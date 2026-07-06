# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import math
import random
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_REPLAY_PRIORITY_EPS = 1e-6


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        replay_config = OmegaConf.select(config, "async_training.replay", default={}) or {}
        self.replay_enabled = bool(replay_config.get("enable", False))
        if self.replay_enabled:
            self.max_queue_size = int(replay_config.get("buffer_size", 0))
            if self.max_queue_size <= 0:
                raise ValueError(f"async_training.replay.buffer_size must be > 0, got: {self.max_queue_size}")
        else:
            if max_queue_size is None:
                raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
            self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)
        self.producer_done = False
        self._rng = random.Random(int(replay_config.get("seed", 0))) if self.replay_enabled else None
        self.replay_alpha = float(replay_config.get("alpha", 0.0)) if self.replay_enabled else 0.0
        self.replay_beta = float(replay_config.get("beta", 0.0)) if self.replay_enabled else 0.0

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={self.max_queue_size}")

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            if self.replay_enabled and sample is None:
                self.producer_done = True
                self._consumer_condition.notify_all()
                return True

            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if self.replay_enabled:
                return True
            if is_drop:
                return False
            return True

    async def get_samples(self, batch_size: int) -> tuple[list[Any], int] | None:
        """
        Get replay samples without removing them from the queue.

        Returns:
            tuple: (samples, queue_length), or None when producer finished before
            a full replay batch became available.
        """
        if not self.replay_enabled:
            raise RuntimeError("get_samples is only available when async_training.replay.enable=True")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {batch_size}")

        async with self._lock:
            while len(self.queue) < batch_size and not self.producer_done and self.running:
                await self._consumer_condition.wait()

            if len(self.queue) < batch_size:
                return None

            queue_snapshot = list(self.queue)
            queue_len = len(queue_snapshot)
            if self.replay_alpha <= 0:
                indices = self._rng.sample(range(queue_len), batch_size)
                samples = [queue_snapshot[idx] for idx in indices]
            else:
                probabilities = self._priority_probabilities(queue_snapshot)
                indices = self._weighted_sample_without_replacement(probabilities, batch_size)
                samples = [queue_snapshot[idx] for idx in indices]

                if self.replay_beta > 0:
                    is_weights = [(1.0 / (queue_len * probabilities[idx])) ** self.replay_beta for idx in indices]
                    max_weight = max(is_weights)
                    if max_weight > 0 and math.isfinite(max_weight):
                        is_weights = [weight / max_weight for weight in is_weights]
                    else:
                        is_weights = [1.0] * len(samples)
                else:
                    is_weights = [1.0] * len(samples)
                samples = [
                    self._attach_replay_is_weight(sample, is_weight)
                    for sample, is_weight in zip(samples, is_weights, strict=True)
                ]
            self.total_consumed += len(samples)
            return samples, queue_len

    def _load_replay_sample(self, sample: Any) -> Any:
        if isinstance(sample, (bytes, bytearray)):
            return ray.cloudpickle.loads(sample)
        return sample

    def _dump_replay_sample(self, original_sample: Any, loaded_sample: Any) -> Any:
        if isinstance(original_sample, (bytes, bytearray)):
            return ray.cloudpickle.dumps(loaded_sample)
        return loaded_sample

    def _get_replay_priority(self, sample: Any) -> float:
        try:
            loaded_sample = self._load_replay_sample(sample)
            priority = float(getattr(loaded_sample, "replay_priority", 1.0))
        except Exception:
            priority = 1.0
        if not math.isfinite(priority) or priority <= 0:
            return _REPLAY_PRIORITY_EPS
        return priority

    def _attach_replay_is_weight(self, sample: Any, weight: float) -> Any:
        try:
            loaded_sample = self._load_replay_sample(sample)
            setattr(loaded_sample, "replay_is_weight", float(weight))
            return self._dump_replay_sample(sample, loaded_sample)
        except Exception:
            return sample

    def _priority_probabilities(self, queue_snapshot: list[Any]) -> list[float]:
        weighted_priorities = [self._get_replay_priority(sample) ** self.replay_alpha for sample in queue_snapshot]
        total = sum(weighted_priorities)
        if total <= 0 or not math.isfinite(total):
            return [1.0 / len(queue_snapshot)] * len(queue_snapshot)
        return [priority / total for priority in weighted_priorities]

    def _weighted_sample_without_replacement(self, probabilities: list[float], batch_size: int) -> list[int]:
        available_indices = list(range(len(probabilities)))
        available_weights = list(probabilities)
        selected_indices = []

        for _ in range(batch_size):
            total_weight = sum(available_weights)
            if total_weight <= 0 or not math.isfinite(total_weight):
                selected_position = self._rng.randrange(len(available_indices))
            else:
                threshold = self._rng.random() * total_weight
                cumulative = 0.0
                selected_position = len(available_weights) - 1
                for position, weight in enumerate(available_weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        selected_position = position
                        break

            selected_indices.append(available_indices.pop(selected_position))
            available_weights.pop(selected_position)

        return selected_indices

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_samples(self, batch_size: int) -> Any | None:
        """Get replay samples from queue without removing them (async)"""
        future = self.queue_actor.get_samples.remote(batch_size)
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
