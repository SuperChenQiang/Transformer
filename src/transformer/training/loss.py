"""TODO：实现忽略 PAD 的交叉熵和标签平滑。

输入原始 logits [B, T, V] 与标签 [B, T]。
明确按有效 token 归一化；不要先对 logits 做 softmax 再传入交叉熵。
"""
