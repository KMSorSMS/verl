#!/usr/bin/env bash
# 8B 视频 V1/TQ separate_async (FSDP2) — 大 payload 数据面对比。默认 SimpleStorage;mooncake 档另配。
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export VLLM_USE_V1=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo max > /sys/fs/cgroup/pids.max 2>/dev/null || true
HF_MODEL_PATH=${HF_MODEL_PATH:-/ufs/models/Qwen3-VL-8B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/ufs/yzw/data/onethinker/onethinker_llava_train.parquet}
TEST_FILE=${TEST_FILE:-/ufs/yzw/data/onethinker/onethinker_llava_val.parquet}
REWARD_FILE=${REWARD_FILE:-/ufs/yzw/verl/onethinker_reward.py}
BSZ=${BSZ:-16}; MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}; MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-32768}; GEN_TP=${GEN_TP:-2}; ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}; MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-20480}
STALENESS=${STALENESS:-1}; N_TRAIN_NODES=${N_TRAIN_NODES:-1}; N_ROLLOUT_NODES=${N_ROLLOUT_NODES:-1}; N_GPUS=${N_GPUS:-8}
TOTAL_STEPS=${TOTAL_STEPS:-10}; RAY_CPUS=${RAY_CPUS:-null}
python3 -m verl.trainer.main_ppo \
  trainer.use_v1=True trainer.v1.trainer_mode=separate_async transfer_queue.enable=True \
  trainer.v1.separate_async.num_warmup_batches=2 trainer.v1.separate_async.parameter_sync_step=4 \
  trainer.v1.sampler.max_off_policy_threshold=${STALENESS} trainer.v1.sampler.max_off_policy_strategy=drop \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS} algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" data.val_files="$TEST_FILE" data.train_batch_size=${BSZ} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.video_key=videos data.filter_overlong_prompts=False data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path=$HF_MODEL_PATH actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=${BSZ} actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} actor_rollout_ref.rollout.max_model_len=${MAX_NUM_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.nnodes=${N_ROLLOUT_NODES} actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS} \
  custom_reward_function.path=$REWARD_FILE custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 trainer.logger='["console","tensorboard"]' trainer.project_name=ot8b_v1async trainer.experiment_name=ot8b_s${STALENESS} \
  trainer.n_gpus_per_node=${N_GPUS} trainer.nnodes=${N_TRAIN_NODES} trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_training_steps=${TOTAL_STEPS} trainer.total_epochs=1 "$@"
