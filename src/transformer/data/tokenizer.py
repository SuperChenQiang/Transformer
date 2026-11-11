"""TODO 01：首先在这里实现 WordTokenizer。

计划接口：fit、encode、decode、save、load，及 vocab_size 属性。
固定特殊词元：PAD=0、UNK=1、BOS=2、EOS=3。
只用训练集建立词表；明确大小写、标点、空白等预处理选择。
编码返回 Python 整数列表；批次补齐由 dataset.py 负责。
参考 docs/implementation_order.md 和 tests/test_tokenizer.py。
"""
