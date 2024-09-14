#!/bin/bash

# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# Define the paths for the tokenizer and the model
TOKEN_PATH=~/models/Llama-2-7b-chat-hf
MODEL_PATH=~/models/Llama-2-7b-chat-hf-xft
#TOKEN_PATH=~/models/Qwen2-7B-Instruct
#MODEL_PATH=~/models/Qwen2-7B-Instruct-xft

SPEC_MODEL_PATH=~/models/TinyLlama-1.1B-Chat-v1.0-xft
SPEC_NUM_TOKENS=4

DATASET_PATH=~/models/ShareGPT_V3_unfiltered_cleaned_split.json
DTYPE=bf16
KV_DTYPE=int8

# param for throughput
BATCHED_TOKENS=32768
NUM_PROMPTS=1000
# param for request rate, requests / s
REQ_RATE=2.0e38
#REQ_RATE=0.02 #2.0e38
# param for latency
BATCHED_SEQS=2

#export ENABLE_KV_TRANS=0
#export ENABLE_TUNED_COMM=0
#export FLASH_ATTN_THRESHOLD=32768

node=$1
nth=$2
#local throughtput(0) / local latency(1) / api server(2) / clent bench(3)
RUN_MODE=$3

BASE_SCRIPTS="--tokenizer ${TOKEN_PATH} \
        --model ${MODEL_PATH} \
	--dtype ${DTYPE} \
	--kv-cache-dtype ${KV_DTYPE} "

echo $BASE_SCRIPTS 

if [ "$RUN_MODE" -eq 0 ];then

SCRIPTS_M="python ${PWD}/benchmark_throughput.py ${BASE_SCRIPTS}
		  --max-num-batched-tokens ${BATCHED_TOKENS} \
        	  --dataset ${DATASET_PATH} \
		  --num-prompts ${NUM_PROMPTS}"
SCRIPTS_S=${SCRIPTS_M}

elif [ "$RUN_MODE" -eq 1 ];then

                  #--speculative_model ${SPEC_MODEL_PATH} \
                  #--num_speculative_tokens ${SPEC_NUM_TOKENS} \
SCRIPTS_M="python ${PWD}/benchmark_latency.py ${BASE_SCRIPTS}
                  --speculative_model ${SPEC_MODEL_PATH} \
                  --num_speculative_tokens ${SPEC_NUM_TOKENS} \
		  --use-v2-block-manager \
		  --num-iters-warmup 3 \
		  --num-iters 1 \
		  --input-len 32 \
		  --output-len 8 \
		  --batch-size ${BATCHED_SEQS}"
SCRIPTS_S=${SCRIPTS_M}

elif [ "$RUN_MODE" -eq 2 ];then

## export no_proxy=localhost
# curl http://localhost:8000/v1/completions \
#  -H "Content-Type: application/json" \
#  -d '{
#  "model": "xft",
#  "prompt": "San Francisco is a",
#  "max_tokens": 512,
#  "temperature": 0
#  }'

# or -d @data.json | jd '.choices[0].text'

SCRIPTS_M="python -m vllm.entrypoints.openai.api_server ${BASE_SCRIPTS}
	          --served-model-name xft \
	          --port 8000 \
		  --max-num-seqs ${BATCHED_SEQS} \
	          --trust-remote-code"

SCRIPTS_S="python -m vllm.entrypoints.slave ${BASE_SCRIPTS}"

elif [ "$RUN_MODE" -eq 3 ];then

SCRIPTS_M="python ${PWD}/benchmark_serving.py --tokenizer ${TOKEN_PATH} \
		  --model xft \
        	  --dataset-path ${DATASET_PATH} \
		  --num-prompts ${NUM_PROMPTS} \
		  --request-rate ${REQ_RATE}"
SCRIPTS_S="NULL"
fi

echo $SCRIPTS_M "  " $SCRIPTS_S 

# Use numactl to bind to appropriate CPU resources
if [ "$node" -eq 1 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C  0-`expr $nth - 1` -l ${SCRIPTS_M}

elif [ "$node" -eq 2 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C  0-`expr $nth - 1` -l ${SCRIPTS_M} : \
        -n 1 numactl -C  `expr $nth`-`expr $nth \* 2 - 1` -l ${SCRIPTS_S}

elif [ "$node" -eq 3 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C 0-`expr $nth - 1` -l ${SCRIPTS_M} : \
        -n 1 numactl -C  `expr $nth`-`expr $nth \* 2 - 1` -l ${SCRIPTS_S} : \
        -n 1 numactl -C  `expr $nth \* 2`-`expr $nth \* 2 + $nth - 1` -l ${SCRIPTS_S}

fi
