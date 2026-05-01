#!/usr/bin/env python3
"""Canonical analyzer for the 11-model corrected dataset.

Reads:
- results/cot_faithfulness_anthropic_*.json (Sonnet 4, Opus 4) — full 3-phase
- results/cot_with_hint_realCoT_*.json (5 OpenAI 4.1/4o models, corrected real-CoT)
- results/cot_no_hint_ablation_*.json (7 older models, no-hint ablation)
- results/cot_newer_*.json (Opus 4.7, Sonnet 4.6, GPT-5, GPT-5-mini) — full 3-phase + no-hint
- results/cot_faithfulness_41_expanded_*.json (Phase 2 data for 4.1 + 4o families)

Produces:
- results/final_corrected_analysis.json with pooled stats, per-model rates,
  provider comparisons, and statistical tests (chi-squared with Yates).

Usage:
    python cot_analyze_v2.py
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# Stat helpers (salvaged from cot_analyze_all.py)
# ============================================================

def fisher_exact_2x2(a, b, c, d):
    """Fisher's exact test for 2x2 contingency table. Returns p-value (two-sided)."""
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]])
        return p
    except ImportError:
        n = a + b + c + d
        r1 = a + b
        r2 = c + d
        c1 = a + c
        c2 = b + d

        def log_factorial(k):
            return sum(math.log(i) for i in range(2, k + 1))

        log_p_obs = (
            log_factorial(r1) + log_factorial(r2) + log_factorial(c1) + log_factorial(c2)
            - log_factorial(n) - log_factorial(a) - log_factorial(b)
            - log_factorial(c) - log_factorial(d)
        )
        p_obs = math.exp(log_p_obs)

        p_value = 0.0
        for i in range(min(r1, c1) + 1):
            j = r1 - i
            k = c1 - i
            l = r2 - k
            if j >= 0 and k >= 0 and l >= 0:
                log_p = (
                    log_factorial(r1) + log_factorial(r2)
                    + log_factorial(c1) + log_factorial(c2)
                    - log_factorial(n) - log_factorial(i) - log_factorial(j)
                    - log_factorial(k) - log_factorial(l)
                )
                p = math.exp(log_p)
                if p <= p_obs + 1e-10:
                    p_value += p
        return min(p_value, 1.0)


def chi_squared_2x2_yates(a, b, c, d):
    """Yates-corrected chi-squared for a 2x2 contingency table. Returns (chi2, p).

    Used as the canonical hypothesis test for hint-vs-no-hint and provider-gap
    comparisons. Falls back to a manual implementation if scipy is unavailable.
    """
    try:
        from scipy.stats import chi2_contingency
        chi2, p, _dof, _exp = chi2_contingency([[a, b], [c, d]], correction=True)
        return chi2, p
    except ImportError:
        n = a + b + c + d
        if n == 0:
            return 0.0, 1.0
        row_totals = [a + b, c + d]
        col_totals = [a + c, b + d]
        observed = [a, b, c, d]
        expected = [
            row_totals[0] * col_totals[0] / n,
            row_totals[0] * col_totals[1] / n,
            row_totals[1] * col_totals[0] / n,
            row_totals[1] * col_totals[1] / n,
        ]
        chi2 = 0.0
        for o, e in zip(observed, expected):
            if e == 0:
                continue
            chi2 += (abs(o - e) - 0.5) ** 2 / e
        # p-value from chi-square distribution (df=1)
        # Use survival function approximation via incomplete gamma
        try:
            from math import erfc, sqrt
            # df=1: p = erfc(sqrt(chi2/2))
            p = erfc(sqrt(chi2 / 2.0))
        except Exception:
            p = float('nan')
        return chi2, p


# ============================================================
# Data loaders
# ============================================================

# Canonical per-model lookup of files
MODEL_FILES = {
    # Anthropic (full 3-phase) — Phase 3 with-hint comes from implicit_test
    "claude-opus-4-20250514": {
        "phase23": "cot_faithfulness_anthropic_claude-opus-4-20250514.json",
        "no_hint": "cot_no_hint_ablation_anthropic_claude-opus-4-20250514.json",
        "with_hint_source": "phase23",  # use implicit_test.caught_error
    },
    "claude-sonnet-4-20250514": {
        "phase23": "cot_faithfulness_anthropic_claude-sonnet-4-20250514.json",
        "no_hint": "cot_no_hint_ablation_anthropic_claude-sonnet-4-20250514.json",
        "with_hint_source": "phase23",
    },
    # OpenAI 4.1 / 4o families — Phase 2 from _41_expanded_, Phase 3 with-hint from real-CoT corrected
    "gpt-4.1": {
        "phase2": "cot_faithfulness_41_expanded_20260416_090130.json",
        "with_hint": "cot_with_hint_realCoT_gpt-4.1.json",
        "no_hint": "cot_no_hint_ablation_openai_gpt-4.1.json",
    },
    "gpt-4.1-mini": {
        "phase2": "cot_faithfulness_41_expanded_20260416_090130.json",
        "with_hint": "cot_with_hint_realCoT_gpt-4.1-mini.json",
        "no_hint": "cot_no_hint_ablation_openai_gpt-4.1-mini.json",
    },
    "gpt-4.1-nano": {
        "phase2": "cot_faithfulness_41_expanded_20260416_090130.json",
        "with_hint": "cot_with_hint_realCoT_gpt-4.1-nano.json",
        "no_hint": "cot_no_hint_ablation_openai_gpt-4.1-nano.json",
    },
    "gpt-4o": {
        "phase2": "cot_faithfulness_41_expanded_20260426_160504.json",
        "with_hint": "cot_with_hint_realCoT_gpt-4o.json",
        "no_hint": "cot_no_hint_ablation_openai_gpt-4o.json",
    },
    "gpt-4o-mini": {
        "phase2": "cot_faithfulness_41_expanded_20260426_160504.json",
        "with_hint": "cot_with_hint_realCoT_gpt-4o-mini.json",
        "no_hint": "cot_no_hint_ablation_openai_gpt-4o-mini.json",
    },
    # New SOTA models — full 3-phase and no-hint in one file
    "claude-opus-4-7": {
        "phase23": "cot_newer_claude-opus-4-7.json",
        "with_hint_source": "phase23",
        "no_hint_source": "phase23",
    },
    "claude-sonnet-4-6": {
        "phase23": "cot_newer_claude-sonnet-4-6.json",
        "with_hint_source": "phase23",
        "no_hint_source": "phase23",
    },
    "gpt-5": {
        "phase23": "cot_newer_gpt-5.json",
        "with_hint_source": "phase23",
        "no_hint_source": "phase23",
    },
    "gpt-5-mini": {
        "phase23": "cot_newer_gpt-5-mini.json",
        "with_hint_source": "phase23",
        "no_hint_source": "phase23",
    },
}

PROVIDER_OF = {
    "claude-opus-4-20250514": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-opus-4-7": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
}


def _load_json(name):
    path = RESULTS_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_with_hint(model, cfg):
    """Returns (caught, n) for with-hint Phase 3 using real-CoT methodology."""
    if cfg.get("with_hint_source") == "phase23":
        data = _load_json(cfg["phase23"])
        # full 3-phase JSON: results[i].implicit_test.caught_error
        n, caught = 0, 0
        for r in data.get("results", []):
            if r.get("model") != model:
                continue
            it = r.get("implicit_test", {})
            if "caught_error" in it:
                n += 1
                if it["caught_error"]:
                    caught += 1
        return caught, n
    # cot_with_hint_realCoT_*.json
    data = _load_json(cfg["with_hint"])
    n, caught = 0, 0
    for r in data.get("results", []):
        if "error_found_with_hint_realCoT" in r:
            n += 1
            if r["error_found_with_hint_realCoT"]:
                caught += 1
    return caught, n


def collect_no_hint(model, cfg):
    """Returns (caught, n) for no-hint Phase 3."""
    if cfg.get("no_hint_source") == "phase23":
        # newer-models JSON has no_hint_results array at top level
        data = _load_json(cfg["phase23"])
        n, caught = 0, 0
        for r in data.get("no_hint_results", []):
            if r.get("model") != model:
                continue
            n += 1
            if r.get("caught_error_no_hint"):
                caught += 1
        return caught, n
    data = _load_json(cfg["no_hint"])
    n, caught = 0, 0
    for r in data.get("results", []):
        if "error_found_no_hint" in r:
            n += 1
            if r["error_found_no_hint"]:
                caught += 1
    return caught, n


def collect_phase2_faith(model, cfg):
    """Returns (faithful_count, n) from Phase 2 (explicit_test classification).

    A trial counts as 'faithful' if classification is 'faithful' or 'faithful_partial'.
    """
    if "phase23" in cfg:
        data = _load_json(cfg["phase23"])
        n, faith = 0, 0
        for r in data.get("results", []):
            if r.get("model") != model:
                continue
            cls = r.get("classification", "")
            n += 1
            if cls in ("faithful", "faithful_partial"):
                faith += 1
        return faith, n
    # OpenAI 4.1 / 4o: phase 2 is in the _41_expanded_ JSON
    data = _load_json(cfg["phase2"])
    n, faith = 0, 0
    for r in data.get("results", []):
        if r.get("model") != model:
            continue
        cls = r.get("classification", "")
        n += 1
        if cls in ("faithful", "faithful_partial"):
            faith += 1
    return faith, n


# ============================================================
# Main analysis
# ============================================================

def analyze():
    per_model = {}
    for model, cfg in MODEL_FILES.items():
        wh_caught, wh_n = collect_with_hint(model, cfg)
        nh_caught, nh_n = collect_no_hint(model, cfg)
        faith, p2_n = collect_phase2_faith(model, cfg)

        per_model[model] = {
            "with_hint_real_cot": (wh_caught / wh_n) if wh_n else 0.0,
            "with_hint_real_cot_n": wh_n,
            "with_hint_real_cot_caught": wh_caught,
            "no_hint": (nh_caught / nh_n) if nh_n else 0.0,
            "no_hint_n": nh_n,
            "no_hint_caught": nh_caught,
            "phase2_faithful": faith,
            "phase2_n": p2_n,
            "phase2_faithfulness_rate": (faith / p2_n) if p2_n else 0.0,
            "provider": PROVIDER_OF[model],
        }

    # Pooled with-hint and no-hint
    total_wh_caught = sum(m["with_hint_real_cot_caught"] for m in per_model.values())
    total_wh_n = sum(m["with_hint_real_cot_n"] for m in per_model.values())
    total_nh_caught = sum(m["no_hint_caught"] for m in per_model.values())
    total_nh_n = sum(m["no_hint_n"] for m in per_model.values())
    total_faith = sum(m["phase2_faithful"] for m in per_model.values())
    total_p2_n = sum(m["phase2_n"] for m in per_model.values())

    pooled = {
        "with_hint": (total_wh_caught / total_wh_n) if total_wh_n else 0.0,
        "with_hint_n": total_wh_n,
        "with_hint_caught": total_wh_caught,
        "no_hint": (total_nh_caught / total_nh_n) if total_nh_n else 0.0,
        "no_hint_n": total_nh_n,
        "no_hint_caught": total_nh_caught,
        "phase2_faithfulness": (total_faith / total_p2_n) if total_p2_n else 0.0,
        "phase2_n": total_p2_n,
        "phase2_faithful": total_faith,
    }

    # Provider pooled
    by_provider = defaultdict(lambda: {
        "with_hint_caught": 0, "with_hint_n": 0,
        "no_hint_caught": 0, "no_hint_n": 0,
        "phase2_faithful": 0, "phase2_n": 0,
    })
    for model, s in per_model.items():
        prov = PROVIDER_OF[model]
        by_provider[prov]["with_hint_caught"] += s["with_hint_real_cot_caught"]
        by_provider[prov]["with_hint_n"] += s["with_hint_real_cot_n"]
        by_provider[prov]["no_hint_caught"] += s["no_hint_caught"]
        by_provider[prov]["no_hint_n"] += s["no_hint_n"]
        by_provider[prov]["phase2_faithful"] += s["phase2_faithful"]
        by_provider[prov]["phase2_n"] += s["phase2_n"]

    provider_summary = {}
    for prov, s in by_provider.items():
        provider_summary[prov] = {
            "with_hint_rate": (s["with_hint_caught"] / s["with_hint_n"]) if s["with_hint_n"] else 0.0,
            "with_hint_n": s["with_hint_n"],
            "with_hint_caught": s["with_hint_caught"],
            "no_hint_rate": (s["no_hint_caught"] / s["no_hint_n"]) if s["no_hint_n"] else 0.0,
            "no_hint_n": s["no_hint_n"],
            "no_hint_caught": s["no_hint_caught"],
            "phase2_faithfulness_rate": (s["phase2_faithful"] / s["phase2_n"]) if s["phase2_n"] else 0.0,
            "phase2_n": s["phase2_n"],
            "phase2_faithful": s["phase2_faithful"],
        }

    # Statistical tests (chi-squared, Yates-corrected)
    tests = {}

    # Hint vs no-hint (pooled across all models)
    chi2, p = chi_squared_2x2_yates(
        total_wh_caught, total_wh_n - total_wh_caught,
        total_nh_caught, total_nh_n - total_nh_caught,
    )
    tests["hint_vs_no_hint_pooled"] = {"chi2": chi2, "p": p}

    # Provider gap, with-hint
    if "anthropic" in provider_summary and "openai" in provider_summary:
        a = provider_summary["anthropic"]
        o = provider_summary["openai"]
        chi2, p = chi_squared_2x2_yates(
            a["with_hint_caught"], a["with_hint_n"] - a["with_hint_caught"],
            o["with_hint_caught"], o["with_hint_n"] - o["with_hint_caught"],
        )
        tests["with_hint_provider_gap"] = {"chi2": chi2, "p": p}

        chi2, p = chi_squared_2x2_yates(
            a["no_hint_caught"], a["no_hint_n"] - a["no_hint_caught"],
            o["no_hint_caught"], o["no_hint_n"] - o["no_hint_caught"],
        )
        tests["no_hint_provider_gap"] = {"chi2": chi2, "p": p}

    output = {
        "n_models": len(per_model),
        "models": sorted(per_model.keys()),
        "per_model": per_model,
        "pooled": pooled,
        "provider": provider_summary,
        "statistical_tests": tests,
    }

    out_path = RESULTS_DIR / "final_corrected_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Console summary
    print("=" * 72)
    print("Per-model rates")
    print("=" * 72)
    print(f"{'Model':<28} {'with_hint':>12} {'no_hint':>10} {'faith':>10}")
    for m in sorted(per_model):
        s = per_model[m]
        print(
            f"{m:<28} {s['with_hint_real_cot']*100:>10.1f}% "
            f"({s['with_hint_real_cot_n']:>3}) {s['no_hint']*100:>7.1f}% "
            f"({s['no_hint_n']:>3}) {s['phase2_faithfulness_rate']*100:>7.1f}%"
        )
    print()
    print("Pooled")
    print(f"  with-hint: {pooled['with_hint']*100:.2f}% ({pooled['with_hint_caught']}/{pooled['with_hint_n']})")
    print(f"  no-hint:   {pooled['no_hint']*100:.2f}% ({pooled['no_hint_caught']}/{pooled['no_hint_n']})")
    print(f"  Phase-2 faithfulness: {pooled['phase2_faithfulness']*100:.2f}% ({pooled['phase2_faithful']}/{pooled['phase2_n']})")
    print()
    print("Statistical tests")
    for name, t in tests.items():
        print(f"  {name}: chi2={t['chi2']:.4f}, p={t['p']:.3e}")

    print(f"\nWrote {out_path}")
    return output


if __name__ == "__main__":
    analyze()
