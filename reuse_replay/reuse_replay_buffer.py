# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Bounded-reuse replay-buffer plugin for verl V1 TransferQueue training."""

import time

import numpy as np
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class ReuseReplayBuffer(ReplayBuffer):
    """Bounded FIFO replay that samples original trajectory keys without removal."""

    def should_add_batch_to_generate(self, global_steps: int, partition_id: str, batch_size: int) -> bool:
        reuse_replay = bool(self.sampler_kwargs.get("reuse_replay", False))
        if partition_id != "train" or not reuse_replay:
            return super().should_add_batch_to_generate(
                global_steps=global_steps,
                partition_id=partition_id,
                batch_size=batch_size,
            )

        buffer_size = int(self.sampler_kwargs.get("buffer_size", batch_size))
        if buffer_size < batch_size:
            raise ValueError(f"ReuseReplayBuffer requires buffer_size >= batch_size, got {buffer_size} < {batch_size}")

        self._sync_metadata_from_transfer_queue()

        trajectory_prompt_uids = {key.split("_", 1)[0] for key in self.partitions[partition_id]}
        retained_prompt_uids = (
            self.finished_keys[partition_id].union(self.failure_keys[partition_id]) & trajectory_prompt_uids
        )
        active_prompt_uids = self.pending_keys[partition_id].union(
            self.running_keys[partition_id],
            retained_prompt_uids,
        )

        return len(active_prompt_uids) < buffer_size + batch_size

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        reuse_replay = bool(self.sampler_kwargs.get("reuse_replay", False))
        if partition_id != "train" or not reuse_replay:
            return super().sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)

        buffer_size = int(self.sampler_kwargs.get("buffer_size", batch_size))
        if buffer_size < batch_size:
            raise ValueError(f"ReuseReplayBuffer requires buffer_size >= batch_size, got {buffer_size} < {batch_size}")

        if not hasattr(self, "_reuse_rng"):
            seed = self.sampler_kwargs.get("seed", None)
            self._reuse_rng = np.random.default_rng(None if seed is None else int(seed))

        metrics: dict[str, float] = {}

        while True:
            self._sync_metadata_from_transfer_queue()
            while not self._has_enough_samples(global_steps, partition_id, batch_size):
                time.sleep(self.poll_interval)
                self._sync_metadata_from_transfer_queue()

            prompt_global_steps = self.prompt_global_steps[partition_id]
            sampleable_prompt_uids = sorted(
                self.finished_keys[partition_id].union(self.failure_keys[partition_id]),
                key=lambda key: prompt_global_steps.get(key, 0),
            )

            trajectory_keys_by_prompt_uid = {uid: [] for uid in sampleable_prompt_uids}
            for key in self.partitions[partition_id]:
                uid = key.split("_")[0]
                if uid in trajectory_keys_by_prompt_uid:
                    trajectory_keys_by_prompt_uid[uid].append(key)

            pool_prompt_uids = [uid for uid in sampleable_prompt_uids if trajectory_keys_by_prompt_uid[uid]]
            evict_prompt_uids = [uid for uid in sampleable_prompt_uids if not trajectory_keys_by_prompt_uid[uid]]
            fifo_evict_count = max(0, len(pool_prompt_uids) - buffer_size)
            evict_prompt_uids.extend(pool_prompt_uids[:fifo_evict_count])

            if self.max_off_policy_strategy == "drop":
                for uid in pool_prompt_uids[fifo_evict_count:]:
                    prompt_step = prompt_global_steps.get(uid, 0)
                    staleness = (global_steps - prompt_step + 1) / self.parameter_sync_step
                    if staleness > self.max_off_policy_threshold:
                        evict_prompt_uids.append(uid)

            if evict_prompt_uids:
                evict_prompt_uids = list(dict.fromkeys(evict_prompt_uids))
                evict_set = set(evict_prompt_uids)
                evict_trajectory_keys = [
                    key for uid in evict_prompt_uids for key in trajectory_keys_by_prompt_uid[uid]
                ]
                tq.kv_clear(
                    partition_id=partition_id,
                    keys=list(dict.fromkeys(evict_prompt_uids + evict_trajectory_keys)),
                )
                metrics["training/reuse_replay/evicted_prompts"] = len(evict_prompt_uids)
                metrics["training/reuse_replay/evicted_trajectories"] = len(evict_trajectory_keys)
                sampleable_prompt_uids = [uid for uid in pool_prompt_uids if uid not in evict_set]
            else:
                sampleable_prompt_uids = pool_prompt_uids

            if len(sampleable_prompt_uids) >= batch_size:
                break

        selected = set(self._reuse_rng.choice(sampleable_prompt_uids, size=batch_size, replace=False).tolist())
        keys, tags = [], []
        for key, tag in self.partitions[partition_id].items():
            if key.split("_")[0] in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        batch.extra_info["reuse_replay"] = True
        return batch, metrics
