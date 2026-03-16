#!/usr/bin/env bash
set -u  # 不用 -e，避免某个 wait/子进程失败导致整个脚本退出
set -o pipefail

echo "M3CV cross-task noisy experiment"

PY=csctNoisy.py
OUT=/mnt/data1/tyl/UserID/logdir/noisy_runs
mkdir -p "$OUT"

gpus=(1 2 3 4 5 6)
MAX_JOBS=18   # 你想开 12 也行，但6张卡上建议 <= 6 或 <= 2*6
models=("EEGNet" "MSNet" "IFNet" "FBMSTSNet")

cross_task_groups=(
  "M3CV_Motor M3CV_Rest"
  "M3CV_Motor M3CV_Steady"
  "M3CV_Motor M3CV_P300"
  "M3CV_Motor M3CV_SSVEP_SA"
  "M3CV_Transient M3CV_Rest"
  "M3CV_Transient M3CV_Steady"
  "M3CV_Transient M3CV_P300"
  "M3CV_Transient M3CV_SSVEP_SA"
  "M3CV_Rest M3CV_P300"
  "M3CV_Rest M3CV_SSVEP_SA"
  "M3CV_Steady M3CV_P300"
  "M3CV_Steady M3CV_SSVEP_SA"
)

# 你现在同时跑了 temporal/spatial/zero/all
# 如果想省资源：只跑 ("all") 即可
noise_types=("temporal" "spatial" "zero" "all")

noise_std_scale=1.0
temporal_ratio=0.5
temporal_mode=middle
session_num=12

job_idx=0
pids=()      # 当前正在运行的pid队列（用于并发控制）
descs=()     # 对应描述（便于打印）
logs=()      # 对应日志文件（便于提示）

wait_one() {
  # 等待队列里最早启动的一个任务结束（兼容旧 bash）
  local pid="${pids[0]}"
  local desc="${descs[0]}"
  local log="${logs[0]}"

  echo "[WAIT] pid=$pid :: $desc"
  wait "$pid"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[DONE] pid=$pid :: $desc"
  else
    echo "[FAIL] pid=$pid rc=$rc :: $desc (see $log)"
  fi

  # 出队（删除第0个）
  pids=("${pids[@]:1}")
  descs=("${descs[@]:1}")
  logs=("${logs[@]:1}")
}

for model in "${models[@]}"; do
  for group in "${cross_task_groups[@]}"; do
    read -ra tasks <<< "$group"
    t0="${tasks[0]}"; t1="${tasks[1]}"
    tag="${t0#M3CV_}_${t1#M3CV_}"

    for noise_type in "${noise_types[@]}"; do
      gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
      job_idx=$((job_idx + 1))

      desc="gpu=$gpu model=$model tasks=$t0,$t1 session=$session_num noise=$noise_type"
      log="$OUT/${tag}_${model}_s${session_num}_noise-${noise_type}.log"

      echo "[RUN] $desc"
      # 关键：用 tee 同时输出到屏幕 + 写日志
      # 每行前缀加上 tag，方便区分不同任务输出
      (
        python -u "$PY" \
          --gpuid "$gpu" \
          --model "$model" \
          --session_num "$session_num" \
          --cross_task "$t0" "$t1" \
          --noise_type "$noise_type" \
          --noise_std_scale "$noise_std_scale" \
          --temporal_ratio "$temporal_ratio" \
          --temporal_mode "$temporal_mode" 2>&1 \
        | sed -u "s/^/[$tag|$model|$noise_type] /"
      ) | tee "$log" &
      pid=$!

      pids+=("$pid")
      descs+=("$desc")
      logs+=("$log")

      # 并发控制：达到上限就等最早的一个结束
      while ((${#pids[@]} >= MAX_JOBS)); do
        wait_one
      done
    done
  done
done

# 等待剩余任务
echo "Waiting remaining ${#pids[@]} jobs..."
while ((${#pids[@]} > 0)); do
  wait_one
done

echo "All done. Logs in $OUT"
