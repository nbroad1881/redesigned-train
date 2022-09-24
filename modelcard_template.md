---
{{ card_data }}
---

# {{ model_id | default("CoolModel") }}

This model is fine-tuned for the Kaggle Feedback Prize 3 Competition. 

## Metrics

```
{{ metrics }}
```

## Weights and Biases

The run is logged at this url: https://wandb.ai/markolas/feedback-prize-3/runs/{{ wandb_run_id }}

## Configuration

```
{{ config }}
```