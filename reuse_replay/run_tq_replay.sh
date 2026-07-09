#!/usr/bin/env bash
# REAL TQ path: main_ppo use_v1 + trainer_mode=separate_async + TransferQueue ReplayBuffer.
# A/B: REPLAY_ENABLE=False -> default ReplayBuffer (staleness-min, consume-once = baseline);
#      REPLAY_ENABLE=True  -> our ReuseReplayBuffer custom_sampler (reuse-without-removal + FIFO-N).
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export VLLM_USE_V1=1
echo max > /sys/fs/cgroup/pids.max 2>/dev/null || true

MODEL=${MODEL:-/ufs/models/Qwen3-0.6B}
DATA_TRAIN=${DATA_TRAIN:-/ufs/yzw/data/math_verl/train.parquet}
DATA_VAL=${DATA_VAL:-/ufs/yzw/data/math_verl/test_500.parquet}
N_ROLLOUT=${N_ROLLOUT:-1}          # inference-worker GPUs (few -> gen bottleneck)
N_TRAIN=${N_TRAIN:-4}              # trainer GPUs
GBS=${GBS:-16}                    # train_batch_size = prompt groups / step
PPO_MINI=${PPO_MINI:-16}
ROLLOUT_N=${ROLLOUT_N:-8}          # GRPO group size G
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_RESP=${MAX_RESP:-2048}
SYNC_STEP=${SYNC_STEP:-4}          # trainer.v1.separate_async.parameter_sync_step
STALENESS=${STALENESS:-4}         # max_off_policy_threshold (model versions)
BUFFER=${BUFFER:-64}              # C3 FIFO buffer size N (prompt groups); replay only
REPLAY_ENABLE=${REPLAY_ENABLE:-False}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
GPU_MEM=${GPU_MEM:-0.55}
RAY_CPUS=${RAY_CPUS:-64}
IMAGE_KEY=${IMAGE_KEY:-}          # set to "images" for multimodal (geo3k) -> gen-bottleneck regime
EXP=${EXP:-tq_smoke}
MM=""; [ -n "$IMAGE_KEY" ] && MM="data.image_key=${IMAGE_KEY} +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 +actor_rollout_ref.model.override_config.attn_implementation=${VL_ATTN:-sdpa}"
export TENSORBOARD_DIR=/ufs/yzw/verl_logs/tb/${EXP}; mkdir -p "$TENSORBOARD_DIR"

# Replay arm plugs in our custom sampler; baseline leaves the default ReplayBuffer.
EXTRA=""
if [ "$REPLAY_ENABLE" = "True" ]; then
  EXTRA="trainer.v1.sampler.custom_sampler.path=reuse_replay/reuse_replay_buffer.py \
trainer.v1.sampler.custom_sampler.name=ReuseReplayBuffer \
+trainer.v1.sampler.sampler_kwargs.reuse_replay=True \
+trainer.v1.sampler.sampler_kwargs.buffer_size=${BUFFER} \
+trainer.v1.sampler.sampler_kwargs.alpha=0.0 \
+trainer.v1.sampler.sampler_kwargs.seed=1"
fi

cd /ufs/yzw/verl_c3 || exit 1
# Clear any stale Ray cluster state (docker restart keeps /tmp, so a crashed run's
# /tmp/ray makes ray.init think a cluster exists -> "existing cluster" ValueError).
ray stop --force >/dev/null 2>&1 || true; rm -rf /tmp/ray /tmp/ray_current_cluster >/dev/null 2>&1 || true
python3 -m verl.trainer.main_ppo \
  trainer.use_v1=True trainer.v1.trainer_mode=separate_async transfer_queue.enable=True \
  trainer.v1.separate_async.num_warmup_batches=2 trainer.v1.separate_async.parameter_sync_step=${SYNC_STEP} \
  trainer.v1.sampler.max_off_policy_threshold=${STALENESS} trainer.v1.sampler.max_off_policy_strategy=drop \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS} \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=${DATA_TRAIN} data.val_files=${DATA_VAL} \
  data.train_batch_size=${GBS} data.max_prompt_length=${MAX_PROMPT} data.max_response_length=${MAX_RESP} \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True ${MM} \
  actor_rollout_ref.model.path=${MODEL} actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=${N_ROLLOUT} \
  trainer.n_gpus_per_node=${N_TRAIN} trainer.nnodes=1 trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' trainer.project_name=tq_replay trainer.experiment_name=${EXP} \
  trainer.save_freq=-1 trainer.test_freq=20 trainer.total_training_steps=${TOTAL_STEPS} trainer.val_before_train=True \
  ${EXTRA} "$@"
