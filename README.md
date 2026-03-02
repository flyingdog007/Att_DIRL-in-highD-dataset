AttentionModule（感知特征提取）
结构： 位置编码 + 单层Transformer Encoder 
子模块配置作用线性嵌入22→64, Linear输入特征升维位置编码shape(1,25,64)时序信息注入Multi-Head Attention4头, d_model=64时间步间交互Feed-Forward64→128→64, ReLU非线性特征变换LayerNorm2层梯度稳定性输出(B,64)交互属性向量 φ_c
权重管理：

IRL热身阶段：激活梯度，参与优化（与RewardNetwork联合)
IRL交替阶段：激活梯度，参与优化（与RewardNetwork联合)
PPO交替阶段：权重冻结，detach输出，阻断梯度

RewardNetwork（奖励函数学习）
输入构成：
φ_scalars (B, 3)      [已×PHI_SCALE]
  ├─ φ_1_scaled: -(Δacc_pred-Δacc_expert)² × 1.0
  ├─ φ_2_scaled: -(Δang_pred)²/5 × 100.0
  └─ φ_3_scaled: -轨迹偏差² × 0.1

φ_c (B, 64)           [来自AttentionModule]

action_norm (B, 2)    [Δacc_norm, Δang_norm，归一化空间]

→ 拼接：[φ_scalars(3), φ_c(64), action(2)] = 69维
网络结构：
69 → 128 (ReLU, Dropout 0.1)
  ↓
128 → 64 (ReLU, Dropout 0.1)
  ↓
64 → 32 (ReLU)
  ↓
32 → 1 (Linear)
→ R(s,a;θ)
关键接口：
pythoncompute_phi_scalars(pred_action, expert_delta, pos_t25, acc_t25, ang_t25, ref_frames)
    → Tensor(B, 3)  # 已乘以PHI_SCALE

forward(phi_scalars, phi_c, action_norm)
    → Tensor(B, 1)  # 标量奖励

irl_loss(R_expert, log_Z, reward_net_params)
    → scalar        # IRL损失 + L2正则
权重管理：

IRL阶段：激活梯度，与AttentionModule联合优化
PPO阶段：权重冻结，计算奖励时必须在no_grad下调用

PPOAgent（策略网络）
Actor（动作策略）
输入： [φ_c(64) + ep_info_norm(4)] = 68维
网络结构：
68 → 128 (Tanh)
  ↓
128 → 64 (Tanh)
  ↓
├─ μ分支：64 → 2 (Linear) → mu
└─ σ分支：64 → 2 (Linear) → log_std [clip到-1.5~1.5]
   
标准差：σ = exp(log_std)

输出分布：N(μ, σ)，经tanh压缩到(-1,1)
关键方法：
pythonforward(actor_input)
    → mu, log_std      # 分布参数

get_action_and_logprob(actor_input)
    → action, log_prob, entropy   # 训练时采样

get_deterministic_action(actor_input)
    → tanh(mu)         # 推理时确定性输出

evaluate_action(actor_input, action)
    → log_prob, entropy # PPO多轮中重新评估固定动作
Jacobian修正：
pythonlog_prob_raw = dist.log_prob(u)
log_prob_corrected = log_prob_raw - log(1 - tanh(u)²)

Critic（价值估计）
输入： [φ_c(64) + ep_info_norm(4)] = 68维
网络结构：
68 → 128 (Tanh)
  ↓
128 → 64 (Tanh)
  ↓
64 → 1 (Linear)
→ V(s)，标量
目标值计算： V_target = R(s,a;θ_reward)（IRL学到的奖励，无bootstrap）

PPOAgent（统一封装）
pythonPPOAgent
  ├─ actor: Actor
  ├─ critic: Critic
  ├─ get_action_train(actor_input) → action, log_prob, entropy
  ├─ get_action_infer(actor_input) → tanh(mu)
  └─ compute_ppo_loss(...) → loss_dict {'loss_ppo', 'loss_clip', 'loss_vf', 'loss_ent'}
