# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        distill_kl_coef = float(self.config.algorithm.distill_kl_coef)
        if distill_kl_coef < 0:
            raise ValueError(f"algorithm.distill_kl_coef must be non-negative, got {distill_kl_coef}")
        if distill_kl_coef == 0:
            return super()._update_actor(batch, metrics)

        distill_topk = int(self.config.actor_rollout_ref.rollout.distill_topk)
        if distill_topk <= 0:
            raise ValueError("algorithm.distill_kl_coef requires actor_rollout_ref.rollout.distill_topk > 0")
        if self.use_teacher_policy:
            raise NotImplementedError("behavior-policy top-k KL cannot be combined with teacher-model distillation")
        actor_strategy = self.config.actor_rollout_ref.actor.strategy
        if actor_strategy not in ("fsdp", "fsdp2", "veomni", "megatron"):
            raise NotImplementedError(
                f"behavior-policy top-k KL is not supported for actor strategy {actor_strategy!r}"
            )
        if self.config.actor_rollout_ref.model.get("use_fused_kernels", False):
            raise NotImplementedError("behavior-policy top-k KL requires actor logits; fused kernels hide them")

        actor_fields = [
            "prompts",
            "responses",
            "input_ids",
            "position_ids",
            "response_mask",
            "loss_mask",
            "old_log_probs",
            "advantages",
            "multi_modal_inputs",
            "teacher_topk_logprobs",
            "teacher_topk_ids",
        ]
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            actor_fields.append("ref_log_prob")
        rollout_correction = self.config.algorithm.get("rollout_correction")
        if (
            rollout_correction
            and not rollout_correction.get("bypass_mode", False)
            and rollout_correction.get("rollout_is") is not None
        ):
            actor_fields.append("rollout_is_weights")
        if self.config.actor_rollout_ref.rollout.enable_rollout_routing_replay:
            actor_fields.append("routed_experts")

        extra_info = dict(batch.extra_info or {})
        extra_info["distill_kl_coef"] = distill_kl_coef
        actor_batch = KVBatchMeta(
            keys=batch.keys,
            tags=batch.tags,
            partition_id=batch.partition_id,
            fields=actor_fields,
            extra_info=extra_info,
        )
        return super()._update_actor(actor_batch, metrics)
