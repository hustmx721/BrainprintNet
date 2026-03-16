#!/usr/bin/env bash
set -euo pipefail

# ===============================
# Config
# ===============================
MAX_JOBS=6

models=("DRFNet" "MCGPnet")
# P300 SSVEP DARNNet
datasets=("M3CV_Rest" "M3CV_Transient" "M3CV_Steady" "M3CV_P300" "M3CV_Motor" "M3CV_SSVEP_SA" "LJ30" "MI" "SSVEP")
gpus=(0 1 2 3 4 5)
gpu_idx=0

pids=()   # ★ 新增：保存后台任务 PID

# ===============================
# Run
# ===============================
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do

    gpu=${gpus[$gpu_idx]}
    gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))

    echo "[INFO] Launching: dataset=${dataset}, model=${model}, gpu=${gpu}"

    python -u main.py \
      --setsplit="${dataset}" \
      --gpuid="${gpu}" \
      --model="${model}" &

    pid=$!
    pids+=("$pid")

    # ---- concurrency control (兼容老 bash) ----
    if [ "${#pids[@]}" -ge "$MAX_JOBS" ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi

  done
done

# wait for all remaining jobs
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[INFO] All jobs finished."
