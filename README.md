# Att_DIRL

基于逆强化学习（IRL）和PPO的自主换道行为模仿系统。通过多头自注意力机制提取驾驶交互特征，学习隐含的驾驶奖励函数，最后训练一个表现人类驾驶风格的换道策略。

## 核心思想

1. **特征提取** - 用Transformer的多头自注意力处理25帧观测序列，得到交互属性向量
2. **奖励学习** - IRL阶段学习驾驶奖励（包含效率、平滑度、轨迹吻合度三个维度）
3. **策略优化** - PPO阶段在学到的奖励下训练Actor-Critic策略网络

采用热身+交替训练的方式：先用15轮预训练IRL，再交替进行200轮IRL+PPO联合优化。

## 文件结构

```
├── actor.py              # Actor网络：动作策略
├── critic.py             # Critic网络：状态价值
├── ppo_agent.py          # PPOAgent封装
├── attention_module.py    # 多头自注意力特征提取
├── reward_network.py      # 奖励函数学习
├── utilities.py           # 辅助函数（归一化、checkpoint等）
├── evaluation.py          # 评估函数（单步&滚动生成）
├── training.py            # 主训练循环
└── config.py              # 全局配置
```

## 快速开始

```python
from training import train

# 开始训练
result = train()

# 查看最终评估结果
print(f"最优验证Loss: {result['val_metrics']['irl_loss']:.4f}")
print(f"滚动评估成功率: {result['rollout_metrics']['success_rate']:.2%}")
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| IRL_WARMUP_EPOCHS | 15 | 热身轮数 |
| TOTAL_ALTER_EPOCHS | 200 | 交替训练轮数 |
| IRL_STEPS_PER_ALTER | 5 | 每轮IRL步数 |
| PPO_STEPS_PER_ALTER | 10 | 每轮PPO步数 |
| REWARD_LR | 5e-4 | IRL学习率 |
| PPO_LR | 3e-4 | PPO学习率 |

## 输出指标

**单步评估**（验证集每个样本的预测误差）
- Speed Bias: 速度预测偏差 (m/s)
- Heading Bias: 角度预测偏差 (°)
- Planning Bias: 轨迹位置偏差 (m)

**滚动生成评估**（完整换道轨迹生成）
- ADE: 平均位移误差 (m)
- FDE: 终点位移误差 (m)
- Success Rate: 换道完成率 (%)

## 引用

如果这个项目对你有帮助，欢迎Star！有问题或建议可以提Issue讨论。
