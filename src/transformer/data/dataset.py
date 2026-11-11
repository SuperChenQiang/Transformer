"""TODO：实现序列数据集和 collate_fn。

先实现合成复制任务，再接真实平行文本。
使用 tokenizer 编码、添加 BOS/EOS、补齐 batch 并创建错位目标标签。
计划输出 src_ids、tgt_input_ids、tgt_labels；张量为 torch.long。
原始数据放在仓库根目录 data/，不要与本源码目录混淆。
"""
