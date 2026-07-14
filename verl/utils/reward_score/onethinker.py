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
"""
Reward scoring function for the OneThinker-train-data dataset.

Scoring scheme
--------------
The model is expected to produce responses in the following format::

    <think>
    ... internal reasoning ...
    </think>
    <answer>
    ... final answer ...
    </answer>

The total score is a weighted combination of:

* **Format reward** (weight ``FORMAT_WEIGHT = 0.1``):
  The response must contain both ``<think>…</think>`` and ``<answer>…</answer>`` blocks.

* **Accuracy reward** (weight ``1 - FORMAT_WEIGHT = 0.9``):
  The extracted answer is compared against the ground truth after normalisation.
  - Exact match (case-insensitive, whitespace-collapsed) → full accuracy score.
  - For multiple-choice questions (ground truth is a single letter A–E), a
    response that *starts with* the correct option letter receives half credit.
  - Otherwise → 0.

For structured/grounding tasks (segmentation, bounding boxes) the ground truth
is a JSON string; the comparison falls back to normalised string equality.
"""

import re


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

FORMAT_WEIGHT: float = 0.1
"""Fraction of the total score awarded for correct response formatting."""

CHOICE_LETTERS: str = "ABCDE"
"""Recognised option labels for multiple-choice questions."""


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Collapse whitespace and lower-case a string for fuzzy comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_tagged(text: str, tag: str) -> str:
    """Return the content inside the last occurrence of ``<tag>…</tag>``, or ``""``."""
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def format_reward(predict_str: str) -> float:
    """Return FORMAT_WEIGHT if the response has the required structural tags, else 0."""
    has_think  = bool(re.search(r"<think>.*?</think>",   predict_str, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", predict_str, re.DOTALL))
    return FORMAT_WEIGHT if (has_think and has_answer) else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """Return the accuracy component of the reward (in range [0, 1 - FORMAT_WEIGHT]).

    Args:
        predict_str:  Full model response string.
        ground_truth: Expected answer extracted during data preprocessing.

    Returns:
        float: Accuracy reward in [0, 1 - FORMAT_WEIGHT].
    """
    acc_weight = 1.0 - FORMAT_WEIGHT

    predicted = _extract_tagged(predict_str, "answer")
    gt_norm   = _normalise(ground_truth)
    pred_norm = _normalise(predicted)

    # ── exact match ───────────────────────────
    if pred_norm == gt_norm:
        return acc_weight

    # ── multiple-choice partial credit ────────
    # Ground truth is a bare option letter (e.g. "B")
    if len(ground_truth.strip()) == 1 and ground_truth.strip().upper() in CHOICE_LETTERS:
        letter = ground_truth.strip().upper()
        # Accept if predicted starts with the correct letter, optionally followed
        # by ". " and the option text (e.g. "B. splash the water." or "B: ...").
        if re.match(rf"^{letter}([.\s:].*)?$", predicted.strip(), re.IGNORECASE):
            return acc_weight * 0.5

    return 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    """Compute the combined format + accuracy reward for a single response.

    This function is the entry point registered in
    ``verl.utils.reward_score.default_compute_score``.

    Args:
        predict_str:  Full model response string (decoded token IDs).
        ground_truth: Ground-truth answer string stored in the parquet dataset.

    Returns:
        float: Score in the range [0.0, 1.0].
    """
    return format_reward(predict_str) + acc_reward(predict_str, ground_truth)
