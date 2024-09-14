from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs)
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.utils import make_tensor_with_pad
from vllm.spec_decode.util import create_sequence_group_output
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from vllm.model_executor.layers.sampler import Sampler
import xfastertransformer

logger = init_logger(__name__)

_PAD_SLOT_ID = -1


@dataclass(frozen=True)
class CPUModelInput(ModelRunnerInputBase):
    """
    Used by the CPUModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    sampling_metadata: Optional["SamplingMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
            cls: Type["CPUModelInput"],
            tensor_dict: Dict[str, Any],
            attn_backend: Optional["AttentionBackend"] = None
    ) -> "CPUModelInput":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class CPUModelRunner(ModelRunnerBase[CPUModelInput]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig] = None,
        kv_cache_dtype: Optional[str] = "fp16",
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # Currently, CPU worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.speculative_config = speculative_config
        self.prompt_adapter_config = prompt_adapter_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.attn_backend = None if True else get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization.
        self.model: xfastertransformer.AutoModel  # Set after init_Model

    def load_model(self) -> None:
        # self.model = get_model(model_config=self.model_config,
        #                        load_config=self.load_config,
        #                        device_config=self.device_config,
        #                        lora_config=self.lora_config,
        #                        parallel_config=self.parallel_config,
        #                        scheduler_config=self.scheduler_config,
        #                        cache_config=self.cache_config)
        logger.info(f"Loading xft model {self.model_config.model}, dtype = {self.model_config.dtype}, KV cache dtype = {self.kv_cache_dtype}")
        # longer context model should be init first, since the bug in xFT
        if self.speculative_config:
            self.draft_model = xfastertransformer.AutoModel.from_pretrained(
                self.speculative_config.draft_model_config.model, self.model_config.dtype, self.kv_cache_dtype
            )
            self.spec_decode_sampler = RejectionSampler(disable_bonus_tokens=False, )
            self._metrics = AsyncMetricsCollector(self.spec_decode_sampler)
        self.model = xfastertransformer.AutoModel.from_pretrained(
            self.model_config.model, self.model_config.dtype, self.kv_cache_dtype
        )
        self.sampler = Sampler()

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[List[int]], torch.Tensor, Optional[AttentionMetadata], List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_inputs_list: List[MultiModalInputs] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = len(prompt_tokens)

            seq_lens.append(seq_len)  # Prompt token num
            input_tokens.append(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, seq_len)))

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                mm_kwargs = self.multi_modal_input_mapper(mm_data)
                multi_modal_inputs_list.append(mm_kwargs)

            # Compute the slot mapping.
            # block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            # start_idx = 0
            # if self.sliding_window is not None:
            #     start_idx = max(0, seq_len - self.sliding_window)

            # for i in range(computed_len, seq_len):
            #     if i < start_idx:
            #         slot_mapping.append(_PAD_SLOT_ID)
            #         continue

            #     block_number = block_table[i //
            #                                self.block_size]  # type: ignore
            #     block_offset = i % self.block_size  # type: ignore
            #     slot = block_number * self.block_size + block_offset
            #     slot_mapping.append(slot)

        num_prompt_tokens = len(input_tokens)

        # input_tokens = torch.tensor(input_tokens,
        #                             dtype=torch.long,
        #                             device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore

        attn_metadata = None if True else self.attn_backend.make_metadata(
            is_prompt=True,
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor([]),
            max_decode_seq_len=0,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([]),
            slot_mapping=slot_mapping,
        )

        multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[AttentionMetadata]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        xft_seq_ids: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)
                xft_seq_ids.append(seq_data.xft_ids)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                # block_table = seq_group_metadata.block_tables[seq_id]
                # block_number = block_table[position // self.block_size]
                # block_offset = position % self.block_size
                # slot = block_number * self.block_size + block_offset
                # slot_mapping.append(slot)

                # if self.sliding_window is not None:
                #     sliding_window_blocks = (self.sliding_window //
                #                              self.block_size)
                #     block_table = block_table[-sliding_window_blocks:]
                # block_tables.append(block_table)

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device).unsqueeze(1)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        # block_tables = make_tensor_with_pad(
        #     block_tables,
        #     pad=0,
        #     dtype=torch.int,
        #     device=self.device,
        # )

        attn_metadata = None if True else self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            num_prefills=0,
            block_tables=torch.tensor([]),
        )
        return (
            input_tokens,
            input_positions,
            xft_seq_ids,
            attn_metadata,
        )

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> CPUModelInput:
        return CPUModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> CPUModelInput:
        multi_modal_kwargs = None
        # xft_seq_ids is None for prompts and xft_max_lens is None for decodes
        xft_seq_ids = None
        xft_max_lens = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, attn_metadata, seq_lens,
             multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions, xft_seq_ids,
             attn_metadata) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = []
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since CPU worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            pin_memory=False,
            generators=self.get_generators(finished_requests_ids))

        if is_prompt:
            xft_max_lens = []
            for i in range(len(sampling_metadata.seq_groups)):
                xft_max_lens.append(sampling_metadata.seq_groups[i].sampling_params.max_tokens + seq_lens[i])
                if self.speculative_config:
                    xft_max_lens[i] += self.speculative_config.num_lookahead_slots * 2

        #import pdb
        #pdb.set_trace()
        if self.speculative_config:
            self.spec_is_prompt = self.model.is_prompt(is_prompt)
        if not self.speculative_config or self.spec_is_prompt:
            xft_seq_ids = self.model.set_input_cb(input_tokens, xft_seq_ids, xft_max_lens).tolist()

        if is_prompt:
            for i in range(len(xft_seq_ids)):
                seq_id = list(seq_group_metadata_list[i].seq_data.keys())[0]
                seq_group_metadata_list[i].seq_data[seq_id].xft_ids = xft_seq_ids[i]

        return CPUModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
        )

    def _prepare_spec_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        lookahead_k: int
    ) -> Tuple[List[int], List[int], List[List[int]], List[List[int]],
                List[int], List[int], List[int]]:
        assert len(seq_group_metadata_list) > 0

        xft_ids: List[int] = []
        xft_draft_ids: List[int] = []
        xft_rect_tokens: List[int] = []
        xft_rejected_n: List[int] = []
        xft_max_lens: List[int] = []
        input_tokens: List[List[int]] = []
        placeholder: List[List[int]] = []
        for i, seq_group in enumerate(seq_group_metadata_list):
            prompt_token_ids = list(list(seq_group.seq_data.values())[0].prompt_token_ids)
            gen_token_ids = list(list(seq_group.seq_data.values())[0].output_token_ids)
            xft_max_lens.append(len(prompt_token_ids) + seq_group.sampling_params.max_tokens + lookahead_k * 2)
            seq_id = list(seq_group.seq_data.keys())[0]
            # seq_id equal to xft_ids
            xft_ids.append(seq_group.seq_data[seq_id].xft_ids)
            xft_draft_id = seq_group.seq_data[seq_id].xft_draft_ids
            #import pdb
            #pdb.set_trace()
            if (xft_draft_id == -1):
                input_tokens.append(prompt_token_ids + gen_token_ids)
            else:
                input_tokens.append([-1])
                xft_draft_ids.append(xft_draft_id)
                if len(gen_token_ids) == 1:
                    placeholder.append([gen_token_ids[-1]])
                else:
                    placeholder.append([-1])

            if len(gen_token_ids) == 1:
                xft_rect_tokens.append(gen_token_ids[-1])
                xft_rejected_n.append(1)
            elif xft_draft_id != -1:
                xft_rect_tokens.append(seq_group.seq_data[seq_id].xft_rect_token)
                xft_rejected_n.append(seq_group.seq_data[seq_id].xft_rejected_n)
            #print("(", i, ") '", seq_id, "' (", seq_group.seq_data[seq_id].xft_ids, " ", xft_draft_id, "): ", len(prompt_token_ids), len(gen_token_ids), flush=True)
        return (
            xft_ids,
            xft_draft_ids,
            input_tokens,
            placeholder,
            xft_rejected_n,
            xft_rect_tokens,
            xft_max_lens,
        )


    # @torch.inference_mode()
    def execute_model(
        self,
        model_input: CPUModelInput,
        kv_caches: Optional[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        seq_groups = model_input.sampling_metadata.seq_groups
        # gen tokens phase for spec infer
        if self.speculative_config:
            # adjust lookahead_k
            lookahead_k = self.speculative_config.num_lookahead_slots

            # only those info on rank 0 matters
            (xft_ids, xft_draft_ids, input_tokens, placeholder,
                xft_rejected_n, xft_rect_tokens, xft_max_lens) = self._prepare_spec_decode(seq_groups, lookahead_k)

            if self.spec_is_prompt:
                #import pdb
                #pdb.set_trace()
                xft_draft_ids = self.draft_model.set_input_cb(input_tokens, xft_draft_ids if len(xft_draft_ids) != 0 else None, xft_max_lens).tolist()
                prop = self.draft_model.get_spec_proposals(1)
                for i, seq_group in enumerate(seq_groups):
                    seq_id = list(seq_group.seq_data.keys())[0]
                    seq_group.seq_data[seq_id].xft_draft_ids = xft_draft_ids[i]
            else:
                #import pdb
                #pdb.set_trace()
                org_ids = xft_draft_ids
                xft_draft_ids = self.draft_model.set_input_cb(input_tokens, xft_draft_ids, xft_max_lens).tolist()

                # new ids element->pos
                ids_dict = dict()
                for i, ids in enumerate(xft_draft_ids):
                    ids_dict[ids] = i
                # ord ids element->pos
                ids_mapping = dict()
                for i, ids in enumerate(org_ids):
                    ids_mapping[i] = ids_dict[ids]
                #print(org_ids, ids_dict, ids_mapping)

                #import pdb
                #pdb.set_trace()
                # xft_rejected_n List[int], xft_rect_tokens List[int]
                proposals = self.draft_model.get_spec_proposals(lookahead_k, xft_rejected_n, xft_rect_tokens)

                xft_ids = self.model.set_input_cb(placeholder, xft_ids, xft_max_lens).tolist()
                # token_ids List[List[int]] (accepted tokens + new one token), proposals List[List[int]]
                xft_rejected_n, xft_rect_tokens = self.model.verify_tokens(lookahead_k, proposals)

                # record info for draft request
                for i, seq_group in enumerate(seq_groups):
                    idx = ids_mapping[i] if ids_mapping.keys() else i
                    seq_id = list(seq_group.seq_data.keys())[0]
                    seq_group.seq_data[seq_id].xft_rect_token = xft_rect_tokens[idx]
                    seq_group.seq_data[seq_id].xft_rejected_n = xft_rejected_n[idx]

                # Only perform sampling in the driver worker.
                if not self.is_driver_worker:
                    return []

                #import pdb
                #pdb.set_trace()
                # Sample the next tokens.
                sampler_output_list: List[SamplerOutput] = []
                for step_index in range(lookahead_k + 1):
                    step_output_token_ids: List[CompletionSequenceGroupOutput] = []
                    for i, seq_group in enumerate(seq_groups):
                        idx = ids_mapping[i] if ids_mapping.keys() else i
                        if step_index < lookahead_k - xft_rejected_n[idx]:
                            token_id = proposals[idx][step_index]
                        elif step_index == lookahead_k - xft_rejected_n[idx]:
                            token_id = xft_rect_tokens[idx]
                        else:
                            token_id = -1

                        # Each sequence may have a different num_logprobs; retrieve it.
                        step_output_token_ids.append(
                            create_sequence_group_output(
                                token_id=token_id,
                                token_id_logprob_rank=-1,
                                token_id_logprob=0.0,
                                seq_id = list(seq_group.seq_data.keys())[0],
                                topk_token_ids=[],
                                topk_logprobs=[]
                            ))
                    sampler_output_list.append(
                        SamplerOutput(outputs=step_output_token_ids))

                maybe_rejsample_metrics = (
                    self._metrics.maybe_collect_rejsample_metrics(lookahead_k))
                if maybe_rejsample_metrics is not None:
                    sampler_output_list[
                        0].spec_decode_worker_metrics = maybe_rejsample_metrics

                    # Log time spent in each stage periodically.
                    # This is periodic because the rejection sampler emits metrics
                    # periodically.
                    self._maybe_log_stage_times(*stage_times)
                return sampler_output_list

        # Compute the logits.
        logits = self.model.forward_cb()

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output = self.sampler(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]

    def free_xft_cache(self, xft_seq_ids:List[int]) -> bool:
        #import pdb
        #pdb.set_trace()
        return self.model.free_seqs(
            torch.tensor(xft_seq_ids, dtype=torch.long, device=self.device)
        )
