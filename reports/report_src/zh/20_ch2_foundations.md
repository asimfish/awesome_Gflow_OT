# 第 2 章 GFlowNet 基础与训练目标

只保留通向主线所需的内容：流的定义与正确性定理、内部流自由度的原文出处、训练目标族及其在流空间中的含义。每篇的完整解读见 `reports/`。

## 2.1 训练目标族一览

| 目标 | 约束粒度 | 关键量 | 与主线的关系 |
|---|---|---|---|
| FM（T00） | 单状态入流 = 出流 | 边流 \(F(s\to s')\) | 直接约束流守恒；有环时比值型不稳定（T19 Thm. 3） |
| DB（T02） | 单条边 \(F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | 状态流 \(F(s)\) | 需显式状态流，O08 的训练目标 Eq. (20) 就是 DB + 状态流正则 |
| TB（T03） | 整条轨迹 \(Z\prod P_F=R\prod P_B\) | \(\log Z\) | \(\log Z\) 是学习出来的 baseline；O08 的 LP 设定里 \(Z\) 已知 |
| SubTB(\(\lambda\))（T05） | 任意子轨迹，\(\lambda^{n-m}\) 加权 | 插值参数 \(\lambda\) | DB 与 TB 的连续插值；O08 的 \(\lambda\) 是流正则系数，不是这个 \(\lambda\) |

零残差时四者指向同一 reward-matching 解族；差别只在梯度的空间尺度与信用传播距离。**它们都不回答「选哪个内部流」。**

## 2.2 T00 · 原始 GFlowNet

@@INCLUDE reports/T00_2106.04399.md SECTIONS=2,5,6,7 DEMOTE=2

## 2.3 T02 · GFlowNet Foundations

@@INCLUDE reports/T02_2111.09266.md SECTIONS=2,5,6,7 DEMOTE=2

## 2.4 T03 · Trajectory Balance

@@INCLUDE reports/T03_2201.13259.md SECTIONS=2,5,6,7 DEMOTE=2

## 2.5 T05 · SubTB(λ)

@@INCLUDE reports/T05_2209.12782.md SECTIONS=2,6,7 DEMOTE=2

## 2.6 T10 · 训练诊断

@@INCLUDE reports/T10_2305.07170.md SECTIONS=2,6,7 DEMOTE=2

