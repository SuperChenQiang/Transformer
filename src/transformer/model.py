"""TODO：组装完整 Transformer。

计划接口：Transformer(nn.Module)，包含 encode、decode 和 forward。
连接源/目标嵌入、Encoder、Decoder 和输出投影。
输入词元形状 [B, S] / [B, T]；输出原始 logits [B, T, V_tgt]。
依赖：config、embeddings、encoder、decoder、masks。
"""
