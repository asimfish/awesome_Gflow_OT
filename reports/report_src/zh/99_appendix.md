# 附录 A 论文清单与中译 QA

@@PAPER_TABLE

# 附录 B 符号与术语

| 符号 / 术语 | 含义 | 首见 |
|---|---|---|
| \(G=(\mathcal S,E)\)、\(s_0\)、\(s_f\) | 状态图、源、汇（吸收态） | T02 |
| \(\mathcal X\)、\(U\) | 终止状态集、O08 中的源状态集 | T02 / O08 |
| \(F(s\to s')\)、\(F(s)\) | 边流、状态流（有环时 = 期望访问次数 × 终止流） | T02 / T36 |
| \(P_F\)、\(P_B\) | 前向 / 后向策略，\(F(s\to s')=F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | T02 |
| \(R(x)\)、\(Z\) | 奖励、配分函数 \(Z=F(s_0)=\sum_xR(x)\) | T00 |
| FM / DB / TB / SubTB(\(\lambda\)) | 流匹配 / 细致平衡 / 轨迹平衡 / 子轨迹平衡 训练目标 | T00 / T02 / T03 / T05 |
| 0-flow、\(H^1_+(G)\) | 终止流为零的守恒流；环空间 | T19 |
| flow explosion | 比值型损失把流无限堆进环 | T19 Thm. 3 |
| \(n_\tau\)、\(\mathbb E[n_\tau]\) | 轨迹步数及其期望；\(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) | T36 |
| 最小流（minimum flow）GFlowNet | 在奖励匹配约束下最小化总流的 GFlowNet | T36 Eq. (11) |
| \(L\)、\(d_G(u,x)\) | 源分布；图最短路（跳数）距离 | O08 |
| \(\Gamma(L,R)\)、\(\Pi\) | 耦合集、传输计划 | O01 |
| Kantorovich 问题 / 对偶势 | \(\min_\Pi\sum c\,\Pi\)；顶点标量 \(\pi\) 满足 \(\pi_{s'}-\pi_s\le1\) | O01 / O08 Thm. 3.3 |
| Beckmann 问题 / 最小费用流 | 以边流为变量、顶点守恒为约束的 OT 等价形式 | O02 / O01 Prop. 6.23 |
| 互补松弛 | 最优流只在紧边上为正：\(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\) | O08 Thm. 3.3 |
| Schrödinger 桥（SB）、IPF / IMF | 熵正则动态传输；迭代比例拟合 / 迭代 Markov 拟合 | O04 / C02 / C03 |
| 熵正则 \(\varepsilon\)、Sinkhorn | \(\varepsilon\mathrm{KL}(\Pi\|\alpha\otimes\beta)\) 正则与其交替缩放算法 | O01 |
| unbalanced OT | 两端总质量不等的传输，边缘以罚项代替硬约束 | C01 |
| TV | 全变差距离，本仓库评估终止分布保真度的金标准 | T10 / O08 |

# 参考文献

@@REFERENCES

