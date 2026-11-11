"""TODO：实现 scaled_dot_product_attention 和 MultiHeadAttention。

顺序：QK^T / sqrt(d_head) → mask → softmax → dropout → 与 V 相乘。
布尔 mask 中 True 表示禁止关注；明确广播维度。
多头投影在 [B, L, D] 与 [B, H, L, D/H] 之间转换。
自注意力和交叉注意力共用接口；Q 与 K/V 的序列长度可以不同。
"""
