# NanoGPT — Incremental Build

A step-by-step implementation of a GPT-style language model trained on a text corpus, built up incrementally from a simple bigram model to a scaled transformer.

## Models

Each file in `models/` adds one concept on top of the previous:

| File | Addition |
|------|----------|
| `1_bigram.py` | Bigram language model |
| `2_w_mlp.py` | MLP layers |
| `3_w_self_attention.py` | Single-head self-attention |
| `4_w_multi_self_attention.py` | Multi-head self-attention |
| `5_w_ffn.py` | Feed-forward network |
| `6_w_blocks.py` | Transformer blocks with residual connections |
| `7_layer_norm.py` | Layer normalization |
| `8_scaling_it.py` | Scaled-up GPT |

## Loss Curves

Training and validation losses for each model are saved in `losses/` as JSON files. Run `plot_losses.py` to generate `loss_curves.png` comparing all models.

## Usage

```bash
python models/<model_file>.py
```

Requires an `input.txt` text file in the root directory as the training corpus.
