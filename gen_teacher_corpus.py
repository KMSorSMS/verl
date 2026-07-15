#!/usr/bin/env python3
# PATCH(offline-kd): Standalone offline sequence-level teacher corpus generator.
"""Generate a fixed response-token/top-k corpus with a vLLM teacher.

The input is a verl-style Parquet prompt dataset. A prompt value may be either
a chat-message list (the usual ``prompt`` column), a string, or an existing
list of token ids. The output is a streaming Parquet file whose wide top-k
columns can later be injected into TransferQueue without running student
rollout generation.

Example (the 0.6B launchers in this fork use this GSM8K dataset layout)::

    python gen_teacher_corpus.py \
      --input "$RAY_DATA_HOME/data/gsm8k/train.parquet" \
      --output /ufs/yzw/data/kd/qwen3_8b_gsm8k_k8000.parquet \
      --teacher-model /models/Qwen3-8B --topk 8000 --tensor-parallel-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TEACHER_MODEL = "/models/Qwen3-8B"
DEFAULT_TOPK = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input verl-style prompt Parquet file.")
    parser.add_argument("--output", required=True, help="Output teacher-corpus Parquet file.")
    parser.add_argument("--prompt-key", default="prompt", help="Input column containing chats/text/token ids.")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL, help="Teacher model path or HF id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path; defaults to --teacher-model.")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Teacher alternatives stored per token.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--truncation", choices=("error", "left", "right"), default="error")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 gives deterministic greedy sequences.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1, help="vLLM sampling top-k, distinct from --topk.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts submitted to vLLM per call/row group.")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N rows (smoke mode).")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--add-generation-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the assistant prefix when applying a chat template.",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help='JSON kwargs for tokenizer.apply_chat_template, e.g. \'{"enable_thinking": false}\'.',
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.topk <= 0:
        raise ValueError(f"--topk must be positive, got {args.topk}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if args.max_prompt_length <= 0:
        raise ValueError(f"--max-prompt-length must be positive, got {args.max_prompt_length}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}")
    if args.temperature < 0:
        raise ValueError(f"--temperature must be non-negative, got {args.temperature}")
    if not 0 < args.top_p <= 1:
        raise ValueError(f"--top-p must be in (0, 1], got {args.top_p}")


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize common tokenizer return shapes to one flat Python list."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"Expected one tokenized prompt, got batch size {len(token_ids)}")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(not isinstance(token_id, int) for token_id in token_ids):
        raise TypeError(f"Prompt tokenization did not produce a flat integer list: {type(token_ids)!r}")
    return token_ids


def maybe_parse_json_chat(prompt: Any) -> Any:
    if not isinstance(prompt, str):
        return prompt
    stripped = prompt.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return prompt
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return prompt
    if isinstance(parsed, list) and (not parsed or isinstance(parsed[0], dict)):
        return parsed
    return prompt


def tokenize_prompt(
    prompt: Any,
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    prompt = maybe_parse_json_chat(prompt)
    if isinstance(prompt, list) and (not prompt or isinstance(prompt[0], int)):
        return normalize_token_ids(prompt)
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        tokenized = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **chat_template_kwargs,
        )
        return normalize_token_ids(tokenized)
    if isinstance(prompt, str):
        return normalize_token_ids(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    raise TypeError(
        "Prompt must be chat messages, text, or token ids; "
        f"got {type(prompt)!r}. Use --prompt-key to select the correct column."
    )


def truncate_prompt(token_ids: list[int], max_length: int, mode: str, source_index: int) -> list[int]:
    if len(token_ids) <= max_length:
        return token_ids
    if mode == "error":
        raise ValueError(
            f"Input row {source_index} has {len(token_ids)} prompt tokens, exceeding "
            f"--max-prompt-length={max_length}"
        )
    if mode == "left":
        return token_ids[-max_length:]
    return token_ids[:max_length]


def iter_prompt_batches(
    input_path: str,
    prompt_key: str,
    batch_size: int,
    limit: int | None,
) -> Iterator[tuple[list[int], list[Any]]]:
    parquet_file = pq.ParquetFile(input_path)
    if prompt_key not in parquet_file.schema_arrow.names:
        raise KeyError(
            f"Prompt column {prompt_key!r} is absent from {input_path}; "
            f"available columns: {parquet_file.schema_arrow.names}"
        )

    seen = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[prompt_key]):
        prompts = batch.column(0).to_pylist()
        if limit is not None:
            prompts = prompts[: max(0, limit - seen)]
        if not prompts:
            break
        indices = list(range(seen, seen + len(prompts)))
        yield indices, prompts
        seen += len(prompts)
        if limit is not None and seen >= limit:
            break


def sampled_token_logprobs(request_output: Any) -> list[float]:
    """Port the rollout's large-k sampled-token lookup fallback."""
    completion = request_output.outputs[0]
    if completion.logprobs is None or len(completion.logprobs) != len(completion.token_ids):
        returned = len(completion.logprobs) if completion.logprobs is not None else None
        raise ValueError(
            "vLLM response logprobs are missing or not aligned with generated token ids: "
            f"got {returned}, expected {len(completion.token_ids)}"
        )

    values = []
    for token_id, logprobs in zip(completion.token_ids, completion.logprobs, strict=True):
        entry = logprobs.get(token_id)
        if entry is not None:
            value = float(entry.logprob)
        elif logprobs:
            # At large k, vLLM may omit the sampled id. Match the online path's
            # conservative fallback so corpus generation does not abort.
            value = min(float(candidate.logprob) for candidate in logprobs.values())
        else:
            value = 0.0
        values.append(value if math.isfinite(value) else 0.0)
    return values


def corpus_schema(topk: int, metadata: dict[str, str]) -> pa.Schema:
    topk_ids = pa.list_(pa.int32(), topk)
    topk_logprobs = pa.list_(pa.float32(), topk)
    return pa.schema(
        [
            pa.field("source_index", pa.int64(), nullable=False),
            pa.field("prompt_token_ids", pa.large_list(pa.int64()), nullable=False),
            pa.field("response_token_ids", pa.large_list(pa.int64()), nullable=False),
            pa.field("teacher_topk_ids", pa.large_list(topk_ids), nullable=False),
            pa.field("teacher_topk_logprobs", pa.large_list(topk_logprobs), nullable=False),
            pa.field("rollout_log_probs", pa.large_list(pa.float32()), nullable=False),
            pa.field("response_mask", pa.large_list(pa.int8()), nullable=False),
            pa.field("loss_mask", pa.large_list(pa.int8()), nullable=False),
            pa.field("prompt_length", pa.int32(), nullable=False),
            pa.field("response_length", pa.int32(), nullable=False),
            pa.field("finish_reason", pa.string()),
            pa.field("stop_reason", pa.string()),
        ],
        metadata={key.encode(): value.encode() for key, value in metadata.items()},
    )


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    """Fingerprint token-to-id compatibility between teacher and student."""
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items()):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    kwargs = {
        "model": args.teacher_model,
        "tokenizer": args.tokenizer or args.teacher_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "max_logprobs": args.topk,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return LLM(**kwargs)


def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        logprobs=args.topk,
    )


def request_prompt(token_ids: Sequence[int]) -> dict[str, list[int]]:
    # A plain mapping is supported across the vLLM versions admitted by this
    # fork (0.8.5 through 0.12.0), unlike version-specific TokensPrompt imports.
    return {"prompt_token_ids": list(token_ids)}


def build_rows(
    source_indices: list[int],
    prompt_token_ids: list[list[int]],
    outputs: Sequence[Any],
    *,
    topk: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    from verl.workers.rollout.vllm_rollout.utils import extract_response_topk_logprobs

    if len(outputs) != len(prompt_token_ids):
        raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(prompt_token_ids)} prompts")

    rows = []
    for source_index, prompt_ids, output in zip(source_indices, prompt_token_ids, outputs, strict=True):
        if not output.outputs:
            raise ValueError(f"vLLM returned no completion for input row {source_index}")
        completion = output.outputs[0]
        response_ids = [int(token_id) for token_id in completion.token_ids]
        if not response_ids:
            raise ValueError(f"Teacher generated an empty response for input row {source_index}")

        topk_fields: dict[str, list] = {}
        extract_response_topk_logprobs(
            output=output,
            topk=topk,
            result_dict=topk_fields,
            vocab_size=vocab_size,
        )
        response_mask = [1] * len(response_ids)
        rows.append(
            {
                "source_index": source_index,
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "teacher_topk_ids": topk_fields["teacher_topk_ids"],
                "teacher_topk_logprobs": topk_fields["teacher_topk_logprobs"],
                "rollout_log_probs": sampled_token_logprobs(output),
                "response_mask": response_mask,
                "loss_mask": response_mask,
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": None if getattr(completion, "stop_reason", None) is None else str(completion.stop_reason),
            }
        )
    return rows


def generate_corpus(args: argparse.Namespace) -> int:
    # Apply before engine construction/generation: decoding 8000 alternative
    # token ids per position can overflow tokenizer decode internals.
    from verl.workers.rollout.vllm_rollout.utils import disable_vllm_logprob_token_decoding

    disable_vllm_logprob_token_decoding()
    llm = make_llm(args)
    tokenizer = llm.get_tokenizer()
    sampling_params = make_sampling_params(args)
    try:
        chat_template_kwargs = json.loads(args.chat_template_kwargs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --chat-template-kwargs JSON: {exc}") from exc
    if not isinstance(chat_template_kwargs, dict):
        raise TypeError("--chat-template-kwargs must decode to a JSON object")

    metadata = {
        "format": "verl_offline_sequence_kd_v1",
        "input": os.path.abspath(os.path.expanduser(args.input)),
        "prompt_key": args.prompt_key,
        "teacher_model": args.teacher_model,
        "tokenizer": args.tokenizer or args.teacher_model,
        "teacher_topk": str(args.topk),
        "vocab_size": str(len(tokenizer)),
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(tokenizer),
        "max_new_tokens": str(args.max_new_tokens),
        "max_prompt_length": str(args.max_prompt_length),
        "truncation": args.truncation,
        "add_generation_prompt": str(args.add_generation_prompt),
        "chat_template_kwargs": json.dumps(chat_template_kwargs, sort_keys=True),
        "temperature": str(args.temperature),
        "top_p": str(args.top_p),
        "sampling_top_k": str(args.top_k),
        "seed": str(args.seed),
    }
    schema = corpus_schema(args.topk, metadata)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it")
    temporary_path = output_path.with_name(f".{output_path.name}.inprogress")
    if temporary_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Incomplete output already exists: {temporary_path}; pass --overwrite to replace it")
        temporary_path.unlink()

    written = 0
    writer = pq.ParquetWriter(temporary_path, schema=schema, compression=args.compression)
    try:
        for source_indices, prompts in iter_prompt_batches(args.input, args.prompt_key, args.batch_size, args.limit):
            prompt_ids = [
                truncate_prompt(
                    tokenize_prompt(
                        prompt,
                        tokenizer,
                        add_generation_prompt=args.add_generation_prompt,
                        chat_template_kwargs=chat_template_kwargs,
                    ),
                    args.max_prompt_length,
                    args.truncation,
                    source_index,
                )
                for source_index, prompt in zip(source_indices, prompts, strict=True)
            ]
            outputs = llm.generate(
                [request_prompt(token_ids) for token_ids in prompt_ids],
                sampling_params,
                use_tqdm=False,
            )
            rows = build_rows(
                source_indices,
                prompt_ids,
                outputs,
                topk=args.topk,
                vocab_size=len(tokenizer),
            )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            written += len(rows)
            print(f"wrote {written} teacher samples", flush=True)
    except BaseException:
        writer.close()
        raise
    else:
        writer.close()

    if written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No prompt rows were read from {args.input}")
    os.replace(temporary_path, output_path)
    print(f"completed {written} samples: {output_path}")
    return written


def main() -> None:
    args = parse_args()
    validate_args(args)
    generate_corpus(args)


if __name__ == "__main__":
    main()
