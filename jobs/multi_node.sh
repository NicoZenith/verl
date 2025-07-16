#!/bin/bash
###############################################################################
#  CSCS – Clariden (A100) – multi‑node GRPO example
#  Ray tmp under $SCRATCH but shortened to avoid AF_UNIX path limit
###############################################################################
#SBATCH -A a135
#SBATCH --job-name=verl-grpo
#SBATCH --nodes=2                     # 2 nodes × 4 GPUs
#SBATCH --ntasks-per-node=1            # one container shell per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10             # tune if needed
#SBATCH --partition=normal             # or debug
#SBATCH --time=01:00:00
#SBATCH --output=job_outputs/%x.out
#SBATCH --exclusive

################# global env #################
set -euo pipefail
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES             # verl wants HIP/CUDA only

# your Verl checkout
export WORK_DIR=$SCRATCH/code/verl
cd "$WORK_DIR"

# small file used to broadcast head address
export SYNC_FILE="$WORK_DIR/.ray_head_addr"
rm -f "$SYNC_FILE"

###############################################################################
#  SHORT Ray session root under scratch (avoid long AF_UNIX paths)
#  Example: /capstor/scratch/cscs/ndeperr/r567890  (short base!)
###############################################################################
export RAY_ROOT_SHORT="${SCRATCH}/r${SLURM_JOB_ID}"

# clean up old if reusing nodes (best effort)
rm -rf "${RAY_ROOT_SHORT}"* 2>/dev/null || true

# Also export a short /tmp fallback in case scratch path still too long
export RAY_ROOT_FALLBACK="/tmp/r${SLURM_JOB_ID}"
rm -rf "${RAY_ROOT_FALLBACK}"* 2>/dev/null || true

###############################################################################
# One *srun* task per node; everything below runs **inside** the Pyxis container
###############################################################################
srun --label --exclusive --environment=verl -N "$SLURM_NNODES" -n "$SLURM_NNODES" bash <<'INNER'
set -euo pipefail
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES

# inherit WORK_DIR, SYNC_FILE, RAY_ROOT_SHORT, RAY_ROOT_FALLBACK from outer env

# install runtime deps inside the verl container
pip install -q nltk

###############################################################################
#  find the container’s IPv4 address  (fallback to getent if needed)
###############################################################################
MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [[ -z "$MY_IP" ]]; then
  MY_IP=$(getent hosts "$(hostname)" | awk '{print $1}')
fi
if [[ -z "$MY_IP" ]]; then
  echo "FATAL: could not determine IPv4 address" >&2
  exit 1
fi

# help Ray trust this IP
export RAY_OVERRIDE_NODE_IP_ADDRESS="$MY_IP"
export RAY_USE_CUSTOM_IP=1
export RAY_DISABLE_IPV6=1

# choose short per-proc Ray temp root
#   /capstor/.../r567890_p0  (head)
#   /capstor/.../r567890_p1  (worker)
THIS_RAY_TMP="${RAY_ROOT_SHORT}_p${SLURM_PROCID}"

# If full path length might still break AF_UNIX, auto-fallback to /tmp
# We approximate: Ray appends ~70 chars; keep base <=35 to be safe
if (( ${#THIS_RAY_TMP} > 35 )); then
  echo "WARNING: Ray temp base path '${THIS_RAY_TMP}' long (${#THIS_RAY_TMP}). Using fallback /tmp." >&2
  THIS_RAY_TMP="${RAY_ROOT_FALLBACK}_p${SLURM_PROCID}"
fi
mkdir -p "$THIS_RAY_TMP"

PORT=6379

# ensure no leftover cluster fragment
ray stop --force >/dev/null 2>&1 || true
sleep 2

if [[ "$SLURM_PROCID" == 0 ]]; then
  ########################  HEAD NODE  ########################
  echo "$(date)  Starting Ray head at ${MY_IP}:${PORT} (tmp=$THIS_RAY_TMP)"
  ray start --head \
            --node-ip-address="${MY_IP}" \
            --port="${PORT}" \
            --num-gpus=4 \
            --num-cpus=${SLURM_CPUS_PER_TASK:-10} \
            --temp-dir="$THIS_RAY_TMP" \
            --disable-usage-stats

  # publish head to sync file (visible to all)
  echo "${MY_IP}:${PORT}" > "$SYNC_FILE"
  echo "Head address written to $SYNC_FILE"

  # let workers come up
  sleep 20
  export RAY_ADDRESS="${MY_IP}:${PORT}"
  export EXPECTED_GPUS=$(( SLURM_NNODES * 4 ))
  export RAY_worker_redirect_output=false

  ###########################################################################
  # Active wait: poll until all GPUs seen
  ###########################################################################
  echo "$(date)  Waiting for $EXPECTED_GPUS GPUs to register with Ray …"
  python - <<'PY'
import os, time, ray, sys
addr = os.environ["RAY_ADDRESS"]
expected = int(os.environ["EXPECTED_GPUS"])
deadline = time.time() + 600   # 10 min timeout
seen = -1
while True:
    try:
        ray.init(address=addr, ignore_reinit_error=True, logging_level="ERROR")
        total = sum(n["Resources"].get("GPU", 0) for n in ray.nodes())
        if total != seen:
            print(f"[ray-wait] GPUs visible: {total}/{expected}", flush=True)
            seen = total
        if total >= expected:
            print("[ray-wait] All GPUs registered.", flush=True)
            break
    except Exception as e:
        print(f"[ray-wait] Ray query failed: {e}", flush=True)
    finally:
        ray.shutdown()
    if time.time() > deadline:
        print("[ray-wait] TIMEOUT waiting for full cluster!", flush=True)
        sys.exit(1)
    time.sleep(2)
PY
  echo "$(date)  GPU registration complete. Launching GRPO training driver."

  ######################## Training Driver ########################
  python -u -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="$WORK_DIR/data/mimic/train.parquet" \
      data.val_files="$WORK_DIR/data/mimic/val.parquet" \
      data.train_batch_size=256 \
      data.max_prompt_length=4096 \
      data.max_response_length=512 \
      data.truncation=right \
      data.filter_overlong_prompts=False \
      data.image_key=images \
      actor_rollout_ref.model.path=/capstor/store/cscs/swissai/a135/RadVLM_project/models/Qwen2.5-VL-7B-CS \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      critic.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
      actor_rollout_ref.rollout.n=8 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      custom_reward_function.path="$WORK_DIR/custom_rewards/radgraph_reward.py" \
      custom_reward_function.name=compute_score \
      trainer.n_gpus_per_node=4 \
      trainer.nnodes=$SLURM_NNODES \
      trainer.logger=[console,wandb] \
      trainer.project_name=mimic_grpo_bleu_multinode \
      trainer.experiment_name=qwen2_5_vl_7b_multinode \
      trainer.save_freq=20 \
      trainer.test_freq=5 \
      trainer.total_epochs=1 \
  |& tee driver.log

else
  ########################  WORKER NODE  ######################
  # wait for head to publish address
  while [[ ! -s "$SYNC_FILE" ]]; do sleep 1; done
  HEAD_ADDR=$(cat "$SYNC_FILE")
  echo "$(date)  Worker joining $HEAD_ADDR (tmp=$THIS_RAY_TMP)"

  ray start --address="$HEAD_ADDR" \
            --node-ip-address="$MY_IP" \
            --num-gpus=4 \
            --num-cpus=${SLURM_CPUS_PER_TASK:-10} \
            --temp-dir="$THIS_RAY_TMP" \
            --disable-usage-stats \
            --block
  # never returns (Ray worker runs forever)
fi
INNER

