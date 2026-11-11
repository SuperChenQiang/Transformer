"""TODO：实现 EncoderLayer 和 Encoder。

每层：自注意力 → Add & Norm → 前馈网络 → Add & Norm。
采用 Post-LN；每个子层的输出先 dropout 再与残差相加。
堆叠层具有独立参数，输出 [B, S, D]。
"""
