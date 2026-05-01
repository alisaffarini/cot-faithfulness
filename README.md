# Is Chain-of-Thought Reasoning Faithful? Corruption Probes and Implicit-Detection Ablations Across Eleven Language Models

**Author:** Anonymous Author, Anonymous Institution

## Key Results

- 11 frontier models (4 Anthropic + 7 OpenAI), 1320 Phase-2 trials, 1320 with-hint Phase-3 trials, 330 no-hint Phase-3 trials.
- **Phase 2 (faithfulness):** pooled 56.1%, range 35-69%. Newer models (Opus 4.7, Sonnet 4.6, GPT-5, GPT-5-mini) cluster at 67-69% faithful; older frontier models 50-59%; smallest models (GPT-4.1-nano, GPT-4o-mini) 35-42%. Faithfulness improves with model generation across both providers.
- **Phase 3 (implicit detection on the model's real Phase-1 CoT):** with-hint detection 4.1% (54/1320), no-hint detection 1.5% (5/330); chi^2 = 4.36, p = 0.037. Per-model with-hint detection ranges 1.7-6.7%; no-hint is 0.0% or 3.3% throughout. Detection is at floor across all 11 models.
- **No provider-level gap on implicit detection** under uniform real-CoT prompting: with-hint chi^2 = 1.43, p = 0.23 (NS); without-hint chi^2 = 0.00, p = 1.00 (NS).
- Frontier LLMs essentially do not spontaneously verify their own reasoning, and a single hint sentence produces only a small absolute increment.
- Methodology: the implicit-detection probe feeds the model its actual multi-paragraph Phase-1 CoT rather than a templated three-step scaffold with the corruption written in as a literal step; the latter design encodes the answer into the prompt and inflates apparent detection by an order of magnitude.

## Models

- **Anthropic (4):** Claude Opus 4, Claude Opus 4.7, Claude Sonnet 4, Claude Sonnet 4.6
- **OpenAI (7):** GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini

## Structure

```
paper/       LaTeX source + style file
results/     Trial-level JSON + analysis (final_corrected_analysis.json)
experiment/  Python experiment + analysis scripts (incl. cot_no_hint_ablation.py)
```

## Citation

Target venue: NeurIPS (anonymous submission)

Code and data: https://anonymous.4open.science/r/cot-faithfulness
