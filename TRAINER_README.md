# Mahjong AI Trainer

Local Chinese Standard Mahjong (MCR / 国标麻将) training prototype.

Based on the upstream project:
https://github.com/Mars-Dingdang/Botzone-Chinese-Standard-Mahjong

Development started from upstream commit `583615a`.

## Features

- Human vs 3 AI players
- Local browser GUI
- AI candidate action ranking
- Model policy preference display
- Model logit decomposition
- Counterfactual comparison between two actions
- Batched GPU rollout over 64 paired possible worlds
- Claim-stage comparison for pass / chi / peng
- Terminal CLI for debugging

## Notes

AI percentages are model policy preferences, not calibrated win probabilities.
Counterfactual rollout is an approximate training aid, not an exact game-theoretic evaluation.

## Run

```bash
conda activate mahjong-ai
python web_gui.py
```

Then open `http://127.0.0.1:8000`.

Current milestone: **v0.1 playable prototype**.
