"""TODO：实现 TokenEmbedding 与 SinusoidalPositionalEncoding。

词元输入 [B, L]，嵌入输出 [B, L, D]。
实现 sqrt(d_model) 缩放；位置编码通过 register_buffer 保存。
验证固定位置数值、奇数维度及最大长度边界。
"""
