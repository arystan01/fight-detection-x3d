vllm serve ${1:-Qwen/Qwen3.5-2B} \
  --port 8000 \
  --max-num-seqs 16 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8 \
  --limit-mm-per-prompt '{"image": 8, "video": 0}'
