"""TODO：先实现 greedy_decode，后续再考虑 beam search。

以 BOS 开始，逐步预测，将生成词元接回输入，遇 EOS 或长度上限停止。
使用 model.eval() 和 torch.no_grad()；先缓存一次 encoder 输出。
明确返回值中是否保留 BOS/EOS，以及 batch 内不同结束时间的处理。
"""
