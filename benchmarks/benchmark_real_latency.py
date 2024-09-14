"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptInputs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

DTYPE_LIST = [
    "auto",
    "half",
    "float16",
    "bfloat16",
    "float",
    "float32",
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]


def main(args: argparse.Namespace):
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_draft_tensor_parallel_size=\
            args.speculative_draft_tensor_parallel_size,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=args.use_v2_block_manager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        otlp_traces_endpoint=args.otlp_traces_endpoint,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0, #if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    input_prompt0 = "Once upon a time, there existed a little girl who liked to have adventures."
    input_prompt1 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color denotes the team owner. But it's something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team's colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem that I had to face in the end was overscoping. I had too much planned, and couldn't get it all done. Content-wise, I had several kinds of pasta planned - Wikipedia is just amazing in that regard, split into several different groups, from small Pastina to huge Pasta al forno. But because of time constraints, I ended up scratching most of them, and ended up with 5 different types of small pasta - barely something to start when talking about the evolution of Pasta. Pastas used in the game. Unfortunately, the macs where never used Which is one of the saddest things about the project, really. It had the framework and the features to allow an endless number of elements in there, but I just didn't have time to draw the rest of the assets needed (something I loved to do, by the way). Other non-obvious features had to be dropped, too. For example, when ordering some pasta, you were supposed to select what kind of sauce you'd"
    input_prompt = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    #import pdb
    #pdb.set_trace()
    input_ids0 = tokenizer(input_prompt0, return_tensors="pt", padding=False).input_ids
    input_ids1 = tokenizer(input_prompt1, return_tensors="pt", padding=False).input_ids
    input_ids = tokenizer(input_prompt, return_tensors="pt", padding=False).input_ids

    real_inputs: List[PromptInputs] = [{
        "prompt_token_ids": input_ids.tolist()[0]
    } for batch in np.arange(args.batch_size)]

    real_inputs: List[PromptInputs] = [{
        "prompt_token_ids": input_ids0.tolist()[0]}, {
        "prompt_token_ids": input_ids.tolist()[0]}]

    #real_inputs: List[PromptInputs] = [{
    #    "prompt_token_ids": input_ids1.tolist()[0]}]

    dummy_inputs = real_inputs

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm_outputs = llm.generate(dummy_inputs,
                             sampling_params=sampling_params,
                             use_tqdm=False)
                # print decoding text.
                for ret in llm_outputs:
                    for out in ret.outputs:
                        print(out.text)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            llm_outputs = llm.generate(dummy_inputs,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            # print decoding text.
            for ret in llm_outputs:
                for out in ret.outputs:
                    print(out.text)
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "."
            ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--speculative-draft-tensor-parallel-size',
                        '-spec-draft-tp',
                        type=int,
                        default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=DTYPE_LIST,
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3', 'fp16', 'int8'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument("--enable-prefix-caching",
                        action='store_true',
                        help="Enable automatic prefix caching")
    parser.add_argument('--use-v2-block-manager', action='store_true')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--otlp-traces-endpoint',
        type=str,
        default=None,
        help='Target URL to which OpenTelemetry traces will be sent.')
    args = parser.parse_args()
    main(args)
