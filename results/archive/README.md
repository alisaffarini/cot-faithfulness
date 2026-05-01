# Archive

Earlier or partial result files preserved for reproducibility.

## Files

- **full_analysis.json** — 5-model summary. Mixes valid Phase-2 faithfulness numbers with bogus Phase-3 detection numbers (the latter from the synthetic-skeleton prompt design). Superseded by `../final_corrected_analysis.json` which uses the canonical real-CoT methodology and 11 models.
- **no_hint_ablation_analysis.json** — Pooled hint-vs-no-hint analysis from a 7-model snapshot. Mixes the buggy synthetic-skeleton with-hint rates with the corrected real-CoT no-hint rates, so the with-hint numbers shown here (97-100%) are not the rates the paper cites. Superseded by `../final_corrected_analysis.json`.
- **cot_faithfulness_41_expanded_20260416_090114.json** — Independent n=360 GPT-4.1-family run (separate from the canonical `_090130` run, ~16 seconds apart but with different API responses). Real paid-for API data; valid Phase-2 robustness sample. Phase-3 fields use the buggy synthetic-skeleton design.
