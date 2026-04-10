# Is Chain-of-Thought Reasoning Faithful? Evidence from Corruption Probes Across Seven Language Models

**Author:** Ali Saffarini, Harvard University

## Key Results

- CoT reasoning is faithful only **37-50%** of the time across 7 frontier models
- Anthropic models detect embedded errors at **2-3x** the rate of OpenAI models (p < 10^-15)
- GPT-4.1-nano has highest decorative CoT rate (41.3%) — smaller models fake reasoning more
- Faithfulness and error detection are **dissociable capabilities** (paradox of unfaithfulness)
- 932 trials across 3 domains (math, logic, commonsense)

## Structure

```
paper/       LaTeX source (NeurIPS format) + style file
results/     Full analysis JSON (932 trials)
experiment/  Python experiment + analysis scripts
```

## Citation

Target venue: ICLR 2027
