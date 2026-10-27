# full-transformer

[English](README.md) | 简体中文

一个使用 PyTorch 基础层与张量运算实现的 decoder-only Transformer 语言模型。本仓库用于手撕练习，重点理解张量形状、注意力、归一化和残差连接。

## 模型结构

```text
Token IDs [B, T]
  -> 词嵌入 [B, T, D]
  -> DecoderBlock × num_layer
  -> 最终 LayerNorm
  -> LM Head (D -> vocab_size)
  -> Logits [B, T, vocab_size]
```

目前包含：

- Pre-Norm decoder block：两个独立的 LayerNorm，以及两条残差连接。
- 手写 LayerNorm：统计均值和方差，完成归一化及可学习的缩放、平移。
- 多头因果自注意力：QKV 投影、对 Q/K 应用 RoPE、因果 mask、合头和输出投影。
- SwiGLU 前馈网络：中间维度为 `3 * D`。

模型输出每个序列位置对下一个 token 的原始词表分数。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| [norm.py](norm.py) | LayerNorm 的统计、归一化与可学习缩放、平移。 |
| [attention.py](attention.py) | 多头因果自注意力及旋转位置编码 RoPE。 |
| [ffn.py](ffn.py) | SwiGLU：gate/up 投影、SiLU 激活、逐元素门控和 down 投影。 |
| [decoder_block.py](decoder_block.py) | 组合归一化、attention、FFN 和残差连接。 |
| [model.py](model.py) | `CZWModel`：词嵌入、多个 decoder block、最终归一化和词表投影。 |

## 当前进度

- 运行依赖：PyTorch。
- 已实现模型前向计算；暂未包含 tokenizer、训练流程、文本生成流程和预训练权重。
- Attention 当前固定为 8 个头。隐藏维度 `D` 需要是 16 的正整数倍，保证每个 head 的特征维度为偶数，以应用 RoPE。
- RoPE 位置表按构造时传入的序列长度 `T` 生成，目前输入需要使用相同长度。
- 当前实现已做静态检查，前向与反向的数值验证尚待完成。
