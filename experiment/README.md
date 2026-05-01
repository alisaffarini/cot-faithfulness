# Experiment Code

Canonical experiment scripts for the 11-model CoT-faithfulness study (real-CoT methodology throughout).

## Scripts

- **cot_paper_expanded.py** — Original GPT-4.1-family driver. Phase 1, Phase 2, and the `classify_faithfulness_fixed` classifier are imported by all canonical scripts. (The script's own `phase3_implicit_test_structured` function is deprecated — see the function docstring; canonical Phase 3 with-hint comes from `cot_with_hint_realCoT.py`.)
- **cot_anthropic.py** — Full 3-phase driver for Anthropic models with real CoTs throughout (Sonnet 4, Opus 4).
- **cot_with_hint_realCoT.py** — Phase 3 with-hint re-run using each model's real Phase-1 CoT (matches the Anthropic methodology). Reuses saved Phase-1 CoTs from existing JSONs; one API call per (model, problem, corruption).
- **cot_no_hint_ablation.py** — Phase 3 no-hint ablation. Reuses saved Phase-1 CoTs; one API call per (model, problem) since the prompt no longer depends on corruption type.
- **cot_newer_models.py** — Unified full-3-phase + no-hint driver for current SOTA models (Opus 4.7, Sonnet 4.6, GPT-5, GPT-5-mini). Real CoTs throughout.
- **cot_analyze_v2.py** — Canonical analyzer that regenerates `../results/final_corrected_analysis.json` from the raw JSONs above.

## Archive

See `archive/` for the older 5-model analyzer and md5-duplicate JSONs.
