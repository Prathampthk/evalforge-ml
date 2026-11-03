"""Judge prompt provenance and provider-reported token/cost accounting, not quality labels."""

import hashlib
import importlib.metadata
import math
import os


def prompt_manifest(metrics) -> dict:
    prompts = {}
    for metric in metrics:
        for name, prompt in metric.get_prompts().items():
            text = prompt.to_string()
            prompts[f"{metric.name}/{name}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "template": text}
    return prompts


def judge_accounting(result) -> dict:
    usage = result.total_tokens()
    values = usage if isinstance(usage, list) else [usage]
    inputs = sum(u.input_tokens for u in values)
    outputs = sum(u.output_tokens for u in values)
    prices = [os.getenv("JUDGE_INPUT_USD_PER_MILLION"), os.getenv("JUDGE_OUTPUT_USD_PER_MILLION")]
    dollars = None
    if all(p is not None for p in prices):
        rates = [float(p) for p in prices if p is not None]
        if any(not math.isfinite(p) or p < 0 for p in rates):
            raise ValueError("Judge prices must be finite nonnegative USD per million tokens")
        if inputs + outputs > 0:
            dollars = (inputs * rates[0] + outputs * rates[1]) / 1_000_000
    return {
        "provider_reported_input_tokens": inputs if inputs + outputs > 0 else None,
        "provider_reported_output_tokens": outputs if inputs + outputs > 0 else None,
        "estimated_judge_usd": dollars,
        "price_usd_per_million": prices,
        "scope": "RAGAS judge calls with reported token usage only; excludes pipeline calls and unreported retries",
        "usage_status": "observed" if inputs + outputs > 0 else "unavailable; not zero-cost evidence",
    }


def judge_identity() -> dict:
    return {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "version": os.getenv("JUDGE_MODEL_VERSION") or "unspecified provider alias; not immutable",
        "ragas_version": importlib.metadata.version("ragas"),
        "temperature": 0.0,
        "human_label_validation": "not measured; no independently human-labeled answer subset available",
        "judge_disagreement": None,
        "false_positive_rate": None,
        "limitations": "LLM judgment is nondeterministic and is not objective ground truth or answer correctness",
    }
