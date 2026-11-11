"""TODO：实现论文中的 warmup / inverse-square-root 学习率调度。

lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))。
定义清楚 step 的起点、更新顺序以及状态保存加载。
"""
