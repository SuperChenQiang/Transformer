"""TODO：实现 PositionwiseFeedForward。

沿每个位置的最后一维执行 Linear(D, d_ff) → ReLU → Linear(d_ff, D)。
输入、输出均为 [B, L, D]；残差和归一化由 encoder/decoder 层负责。
"""
