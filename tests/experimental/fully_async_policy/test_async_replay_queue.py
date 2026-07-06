# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import importlib
from pathlib import Path
import sys
import types

import pytest


class _RayStub(types.SimpleNamespace):
    def remote(self, *args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda obj: obj


class _OmegaConfStub:
    @staticmethod
    def select(config, key, default=None):
        value = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


class _Sample:
    def __init__(self, name, priority=1.0):
        self.name = name
        self.replay_priority = priority
        self.replay_is_weight = 1.0


@pytest.fixture()
def message_queue_cls(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", _RayStub())
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        types.SimpleNamespace(DictConfig=dict, OmegaConf=_OmegaConfStub),
    )
    module_path = Path(__file__).resolve().parents[3] / "verl/experimental/fully_async_policy/message_queue.py"
    spec = importlib.util.spec_from_file_location("message_queue_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MessageQueue


def _config(buffer_size=4, seed=0, alpha=0.0, beta=0.0):
    return {
        "async_training": {
            "replay": {
                "enable": True,
                "buffer_size": buffer_size,
                "seed": seed,
                "train_steps": 8,
                "alpha": alpha,
                "beta": beta,
            }
        }
    }


def test_fifo_keep_freshest_n(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=3), max_queue_size=99)
        for sample in ["s0", "s1", "s2", "s3", "s4"]:
            assert await queue.put_sample(sample) is True

        assert list(queue.queue) == ["s2", "s3", "s4"]
        assert queue.dropped_samples == 2

    asyncio.run(run())


def test_sample_without_removal_reuses_items_across_calls(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=5, seed=0), max_queue_size=99)
        for sample in range(5):
            await queue.put_sample(sample)

        first, first_len = await queue.get_samples(3)
        second, second_len = await queue.get_samples(3)

        assert first_len == 5
        assert second_len == 5
        assert len(queue.queue) == 5
        assert set(first) & set(second)

    asyncio.run(run())


def test_no_within_batch_duplicates(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=6, seed=3), max_queue_size=99)
        for sample in range(6):
            await queue.put_sample(sample)

        batch, _ = await queue.get_samples(4)

        assert len(batch) == 4
        assert len(set(batch)) == 4

    asyncio.run(run())


def test_prioritized_sampling_draws_high_priority_more_often(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=2, seed=11, alpha=1.0), max_queue_size=99)
        await queue.put_sample(_Sample("low", priority=1.0))
        await queue.put_sample(_Sample("high", priority=9.0))

        counts = {"low": 0, "high": 0}
        for _ in range(1000):
            batch, queue_len = await queue.get_samples(1)
            assert queue_len == 2
            counts[batch[0].name] += 1

        assert counts["high"] > counts["low"] * 4
        assert len(queue.queue) == 2

    asyncio.run(run())


def test_replay_is_weights_are_inverse_probability_and_normalized(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=2, seed=0, alpha=1.0, beta=1.0), max_queue_size=99)
        await queue.put_sample(_Sample("low", priority=1.0))
        await queue.put_sample(_Sample("high", priority=4.0))

        batch, _ = await queue.get_samples(2)
        weights = {sample.name: sample.replay_is_weight for sample in batch}

        assert weights["low"] == pytest.approx(1.0)
        assert weights["high"] == pytest.approx(0.25)

    asyncio.run(run())


def test_alpha_zero_beta_zero_stays_uniform_with_unit_weights(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=3, seed=7, alpha=0.0, beta=0.0), max_queue_size=99)
        samples = [_Sample("s0", priority=1.0), _Sample("s1", priority=5.0), _Sample("s2", priority=25.0)]
        for sample in samples:
            await queue.put_sample(sample)

        counts = {sample.name: 0 for sample in samples}
        for _ in range(1200):
            batch, _ = await queue.get_samples(1)
            assert batch[0].replay_is_weight == 1.0
            counts[batch[0].name] += 1

        assert max(counts.values()) - min(counts.values()) < 100

    asyncio.run(run())


def test_prioritized_sampling_has_no_within_batch_duplicates(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=6, seed=3, alpha=1.0), max_queue_size=99)
        for idx in range(6):
            await queue.put_sample(_Sample(f"s{idx}", priority=idx + 1))

        batch, _ = await queue.get_samples(4)
        names = [sample.name for sample in batch]

        assert len(names) == 4
        assert len(set(names)) == 4
        assert len(queue.queue) == 6

    asyncio.run(run())


def test_producer_done_sentinel_is_not_buffered_and_too_small_batch_returns_none(message_queue_cls):
    async def run():
        queue = message_queue_cls(_config(buffer_size=4, seed=0), max_queue_size=99)
        for sample in ["s0", "s1", "s2"]:
            await queue.put_sample(sample)
        await queue.put_sample(None)

        batch, queue_len = await queue.get_samples(3)
        assert len(batch) == 3
        assert queue_len == 3
        assert None not in queue.queue
        assert list(queue.queue) == ["s0", "s1", "s2"]

        too_small = message_queue_cls(_config(buffer_size=4, seed=0), max_queue_size=99)
        await too_small.put_sample("tail")
        await too_small.put_sample(None)

        assert await too_small.get_samples(2) is None
        assert list(too_small.queue) == ["tail"]

    asyncio.run(run())
