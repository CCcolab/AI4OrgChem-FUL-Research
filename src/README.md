# 核心实现说明

validation.py和adaptive_virtual_rank.py保存已使用的科学函数，direct_jk.py保存直接J/K helper。

当前公开的数据复核入口为../scripts/verify_saved_state.py，依赖Python与NumPy，对五份规范态执行保存数组算术检查，覆盖密度、能量、OV、身份和实际空谱。具体命令见[数值复核与原始数据](../docs/reproducibility.zh-CN.md)。

重新求解源SCF时，按选定构造准备完整输入、实际后端和运行预算。本仓库提供当前保存数值态及其查阅、复核材料。
