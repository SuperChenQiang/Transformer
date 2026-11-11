# 参考资料与借鉴范围

本项目的数学结构以原论文为依据，工程划分参考以下公开项目。
目前骨架由本项目编写，没有复制参考项目中的模型实现。

| 资料 | 用途 |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 模型、损失和优化器设计依据 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 公式讲解、复制任务和注意力可视化 |
| [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) | 模块组织及训练/翻译流程参考 |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | 配置、模型、测试、文档的职责分离 |
| [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer) | 分层目录阅读参考 |

注意 Annotated Transformer 的残差实现采用 Pre-LN，而原论文描述 Post-LN。
本项目第一版采用 Post-LN；参考项目之间的差异需要对照论文核实。

如果后续直接复用第三方代码，应在对应文件标明来源并保留适用的许可声明。
