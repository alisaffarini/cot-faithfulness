# Provenance — CoT Faithfulness

Raw experiment files from the burn-tokens paper directory.

## Files

| File | Description |
|------|-------------|
| `original_full_analysis.json` | Original aggregate analysis (938 trials across 9 models) |
| `original_cot_paper_expanded.py` | OpenAI experiment script (GPT-4.1 models) |
| `original_cot_analyze_all.py` | Analysis/aggregation script |

## Notes

- **Raw trial-level data was not preserved** — only the aggregate `full_analysis.json` survives
- The experiment script only uses the OpenAI SDK. Claude models (Sonnet 4, Opus 4) were tested separately but the Anthropic experiment script was not committed. A replacement `cot_anthropic.py` has been added to `experiment/`.
- 938 total trials includes 6 parse_failure trials from claude-3-5-haiku (2) and claude-haiku-4 (4). Paper reports 932 valid trials from 7 models.
- No CoT experiment run directory exists in burn-tokens/research/runs/ (run 095 was not saved)
