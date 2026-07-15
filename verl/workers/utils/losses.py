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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def _align_response_topk_to_sequence(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Causally align response-token teacher distributions with full-sequence logits."""
    teacher_ids = data["teacher_topk_ids"]
    teacher_logprobs = data["teacher_topk_logprobs"]
    if teacher_ids.is_nested != teacher_logprobs.is_nested:
        raise ValueError("behavior-policy top-k ids and logprobs must use the same tensor layout")

    def sequence_lengths(tensor: torch.Tensor) -> list[int]:
        return tensor.offsets().diff().tolist() if tensor.is_nested else [tensor.shape[1]] * tensor.shape[0]

    prompt_lens = sequence_lengths(data["prompts"])
    response_lens = sequence_lengths(data["responses"])
    sequence_lens = sequence_lengths(data["input_ids"])
    aligned_ids, aligned_logprobs = [], []
    for prompt_len, response_len, sequence_len, ids, logprobs in zip(
        prompt_lens,
        response_lens,
        sequence_lens,
        teacher_ids.unbind(),
        teacher_logprobs.unbind(),
        strict=True,
    ):
        if prompt_len < 1 or sequence_len != prompt_len + response_len:
            raise ValueError(
                f"invalid prompt/response lengths for top-k alignment: {prompt_len=}, {response_len=}, {sequence_len=}"
            )
        if ids.shape != logprobs.shape or ids.shape[0] != response_len:
            raise ValueError(
                "behavior-policy top-k tensors must have response_len rows, got "
                f"ids={tuple(ids.shape)}, logprobs={tuple(logprobs.shape)}, {response_len=}"
            )

        topk = ids.shape[-1]
        prefix_ids = torch.zeros((prompt_len - 1, topk), dtype=ids.dtype, device=ids.device)
        suffix_ids = torch.zeros((1, topk), dtype=ids.dtype, device=ids.device)
        pad_logprob = torch.finfo(logprobs.dtype).min
        prefix_logprobs = torch.full(
            (prompt_len - 1, topk), pad_logprob, dtype=logprobs.dtype, device=logprobs.device
        )
        suffix_logprobs = torch.full((1, topk), pad_logprob, dtype=logprobs.dtype, device=logprobs.device)
        aligned_ids.append(torch.cat((prefix_ids, ids, suffix_ids), dim=0))
        aligned_logprobs.append(torch.cat((prefix_logprobs, logprobs, suffix_logprobs), dim=0))

    return (
        torch.nested.as_nested_tensor(aligned_ids, layout=torch.jagged),
        torch.nested.as_nested_tensor(aligned_logprobs, layout=torch.jagged),
    )


def _compute_behavior_topk_kl(config: ActorConfig, data: TensorDict, student_logits, data_format: str):
    teacher_ids, teacher_logprobs = _align_response_topk_to_sequence(data)
    if config.strategy in ("fsdp", "fsdp2", "veomni"):
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

        teacher_ids = teacher_ids.values().unsqueeze(0)
        teacher_logprobs = teacher_logprobs.values().unsqueeze(0)
        if get_ulysses_sequence_parallel_world_size() > 1:
            teacher_ids = slice_input_tensor(teacher_ids, dim=1)
            teacher_logprobs = slice_input_tensor(teacher_logprobs, dim=1)
        if teacher_ids.shape[:2] != student_logits.shape[:2]:
            raise ValueError(
                f"teacher/student sequence shapes do not match: {teacher_ids.shape[:2]} vs {student_logits.shape[:2]}"
            )
        student_logprobs = F.log_softmax(student_logits.float(), dim=-1)
        student_topk_logprobs = torch.gather(student_logprobs, dim=-1, index=teacher_ids.long())
        teacher_logprobs = teacher_logprobs.float()
        kl = (teacher_logprobs.exp() * (teacher_logprobs - student_topk_logprobs)).sum(dim=-1)
    elif config.strategy == "megatron":
        from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine
        from verl.trainer.distillation.megatron.losses import _VocabParallelKLDivergence

        preprocess = preprocess_thd_engine if data_format == "thd" else preprocess_bshd_engine
        teacher_logprobs, *_ = preprocess(teacher_logprobs, pre_process=True)
        teacher_ids, *_ = preprocess(teacher_ids, pre_process=True)
        if teacher_ids.shape[:2] != student_logits.shape[:2]:
            raise ValueError(
                f"teacher/student sequence shapes do not match: {teacher_ids.shape[:2]} vs {student_logits.shape[:2]}"
            )
        kl, *_ = _VocabParallelKLDivergence.apply(student_logits, teacher_logprobs, teacher_ids, None)
    else:
        raise NotImplementedError(
            f"behavior-policy top-k distillation is not supported for actor strategy {config.strategy!r}"
        )
    return {"behavior_distill_kl": kl}


def behavior_policy_distillation_ppo_loss(
    config: ActorConfig,
    model_output=None,
    data: TensorDict = None,
    dp_group=None,
    student_logits=None,
    data_format: str = "thd",
):
    """Add behavior-policy response top-k forward KL to PPO using the same actor forward pass."""
    if student_logits is not None:
        return _compute_behavior_topk_kl(config, data, student_logits, data_format)

    policy_loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    distill_kl_coef = float(tu.get_non_tensor_data(data, "distill_kl_coef", 0.0))
    if distill_kl_coef <= 0:
        return policy_loss, metrics

    if "behavior_distill_kl" not in model_output:
        raise RuntimeError("behavior-policy top-k KL was requested but the actor engine did not expose logits")
    distill_kl = no_padding_2_padding(model_output["behavior_distill_kl"], data)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    distill_kl_loss = agg_loss(
        loss_mat=distill_kl,
        loss_mask=response_mask.bool(),
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    policy_loss += distill_kl_coef * distill_kl_loss
    metrics["actor/distill_kl_loss"] = Metric(value=distill_kl_loss, aggregation=AggregationType.SUM)
    metrics["actor/distill_kl_coef"] = distill_kl_coef
    return policy_loss, metrics


# PATCH(offline-kd): Pure response-token KD objective for fixed large-teacher corpora.
def offline_sequence_distillation_loss(
    config: ActorConfig,
    model_output=None,
    data: TensorDict = None,
    dp_group=None,
    student_logits=None,
    data_format: str = "thd",
):
    """Compute only response-masked, token-mean teacher top-k forward KL."""
    if student_logits is not None:
        return _compute_behavior_topk_kl(config, data, student_logits, data_format)

    if "behavior_distill_kl" not in model_output:
        raise RuntimeError("offline top-k KD was requested but the actor engine did not expose logits")

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    distill_kl = no_padding_2_padding(model_output["behavior_distill_kl"], data)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    distill_kl_loss = agg_loss(
        loss_mat=distill_kl,
        loss_mask=response_mask.bool(),
        loss_agg_mode="token-mean",
        **config.global_batch_info,
    )
    metrics = {
        "offline_distill_kl_loss": Metric(value=distill_kl_loss, aggregation=AggregationType.SUM),
    }
    return distill_kl_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
