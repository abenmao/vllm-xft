#!/bin/bash

# Preload libiomp5.so by following cmd or LD_PRELOAD=libiomp5.so manually
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# Define the paths for the tokenizer and the model
TOKEN_PATH=~/models/Llama-2-7b-chat-hf
MODEL_PATH=~/models/Llama-2-7b-chat-hf-xft
#TOKEN_PATH=~/models/Qwen2.5-7B-Instruct
#MODEL_PATH=~/models/Qwen2.5-7B-Instruct-xft
#TOKEN_PATH=~/models/Qwen2-72B-Instruct
#MODEL_PATH=~/models/Qwen2-72B-Instruct-xft

SPEC_MODEL_PATH=~/models/TinyLlama-1.1B-Chat-v1.0-xft
#SPEC_MODEL_PATH=~/models/Qwen2.5-0.5B-Instruct-xft
#SPEC_MODEL_PATH=~/models/Qwen2-7B-Instruct-xft

DATASET_PATH=~/downloads/ShareGPT_V3_unfiltered_cleaned_split.json

while [ -n "$1" ]; do
    case $1 in
    -d | --dtype)
        DTYPE=$2
	shift 2
	;;
    -kvd | --kv_cache_dtype)
        KV_TYPE=$2
	shift 2
	;;
    -n | --nthreads)
        NTH=$2
	shift 2
	;;
    -spec | --spec_infer)
        SPEC_ENABLE=$2
	shift 2
	;;
    -spec_n | --spec_num_tokens)
        SPEC_NUM_TOKENS=$2
	shift 2
	;;
    -mod | --mode)
        RUN_MODE=$2
	shift 2
	;;
    -s | --sockets)
        case $2 in
        "0" | "1" | "2" | "3")
            node=$2
	    shift 2
	    ;;
	*)
            Error "sockets must in 0, 1, 2 or 3."
            exit 1
            ;;
        esac
        ;;
    esac
done

DTYPE=${DTYPE:-bf16}
KV_DTYPE=${KV_TYPE:-int8}

# param for throughput
BATCHED_TOKENS=8192 #32768
NUM_PROMPTS=1000
# param for request rate, requests / s
REQ_RATE=2.0e38
#REQ_RATE=0.02 #2.0e38
# param for latency
BATCHED_SEQS=64

#export ENABLE_KV_TRANS=0
#export ENABLE_TUNED_COMM=0
#export FLASH_ATTN_THRESHOLD=32768

# enable spec infer or not
SPEC_ENABLE=${SPEC_ENABLE:-'1'}
# num of threads
nth=${NTH:-'40'}
# local throughtput(0) / local latency(1) / api server(2) / clent bench(3)
RUN_MODE=${RUN_MODE:-'1'}
# num of ranks
node=${node:-'1'}
# num of lookahead tokens
SPEC_NUM_TOKENS=${SPEC_NUM_TOKENS:-'4'}

BASE_SCRIPTS="--tokenizer ${TOKEN_PATH} \
        --model ${MODEL_PATH} \
        --use-v2-block-manager \
        --dtype ${DTYPE} \
        --kv-cache-dtype ${KV_DTYPE} "

if [ "$SPEC_ENABLE" -eq 1 ];then
BASE_SCRIPTS="$BASE_SCRIPTS \
	--speculative_model ${SPEC_MODEL_PATH} \
        --num_speculative_tokens ${SPEC_NUM_TOKENS} "
fi

if [ "$RUN_MODE" -eq 0 ];then

SCRIPTS_M="python ${PWD}/benchmark_throughput.py ${BASE_SCRIPTS}
          --max-num-batched-tokens ${BATCHED_TOKENS} \
          --dataset ${DATASET_PATH} \
          --num-prompts ${NUM_PROMPTS}"
SCRIPTS_S=${SCRIPTS_M}

elif [ "$RUN_MODE" -eq 1 ];then

SCRIPTS_M="python ${PWD}/benchmark_real_latency.py ${BASE_SCRIPTS}
          --num-iters-warmup 2 \
          --num-iters 1 \
          --output-len 256 \
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
