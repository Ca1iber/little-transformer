# full-transformer

English | [简体中文](README.zh-CN.md)

A small decoder-only Transformer language model built with basic PyTorch layers and tensor operations. This is a hands-on learning project focused on understanding tensor shapes, attention, normalization, and residual connections.

## Architecture

```text
Token IDs [B, T]
  -> Token embedding [B, T, D]
  -> DecoderBlock × num_layer
  -> Final LayerNorm
  -> LM head (D -> vocab_size)
  -> Logits [B, T, vocab_size]
```

The model includes:

- Pre-Norm decoder blocks with two independent LayerNorm modules and two residual connections.
- A manual LayerNorm implementation with learnable scale and bias.
- Causal multi-head self-attention, including QKV projection, RoPE on Q/K, head concatenation, and output projection.
- A SwiGLU feed-forward network with an intermediate width of `3 * D`.

The output contains raw next-token scores at each sequence position.

## Files

| File | Responsibility |
| --- | --- |
| [norm.py](norm.py) | LayerNorm statistics, normalization, and learnable scale/bias. |
| [attention.py](attention.py) | Multi-head causal self-attention and rotary position embeddings. |
| [ffn.py](ffn.py) | SwiGLU: gate/up projections, SiLU activation, elementwise gating, and down projection. |
| [decoder_block.py](decoder_block.py) | Combines normalization, attention, FFN, and residual connections. |
| [model.py](model.py) | `CZWModel`: token embedding, stacked decoder blocks, final normalization, and vocabulary projection. |

## Current scope

- Runtime dependency: PyTorch.
- The model forward pass is implemented. Tokenization, training, text generation, and pretrained weights are not included yet.
- Attention currently uses 8 heads. The hidden size `D` must be a positive multiple of 16 so that each head has an even feature dimension for RoPE.
- The RoPE table is built for the constructor's sequence length `T`; inputs currently need to use that same length.
- The implementation has been statically reviewed; numerical forward/backward validation is still pending.
