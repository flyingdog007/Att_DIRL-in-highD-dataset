import os
import json
import time
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from config import (
    ENDPOINT_MODEL_PATH, ATT_DIRL_MODEL_PATH, ATT_DIRL_CKPT_DIR, LOG_DIR,
    IRL_WARMUP_EPOCHS, IRL_STEPS_PER_ALTER, PPO_STEPS_PER_ALTER,
    TOTAL_ALTER_EPOCHS, REWARD_LR, PPO_LR, PPO_UPDATE_EPOCHS, PPO_GRAD_CLIP,
    EVAL_INTERVAL, CKPT_SAVE_INTERVAL, CKPT_KEEP_LAST_N, ROLLOUT_EVAL_N,
)
from data_loader import build_all_datasets_and_loaders
from endpoint_predictor import EndpointPredictor
from attention_reward import AttentionModule, RewardNetwork, compute_log_Z_uniform
from ppo_agent import PPOAgent
from utilities import (
    normalize_ep_info, ensure_tensor, batch_to_device,
    save_checkpoint, load_checkpoint, cleanup_old_checkpoints, find_latest_checkpoint
)
from evaluation import (
    evaluate_single_step, evaluate_rollout, evaluate_rollout_v2, print_metrics_table
)


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    print("\n[INFO] 构建数据集...")
    data = build_all_datasets_and_loaders()
    main_train_loader = data['main_train_loader']
    main_val_loader = data['main_val_loader']
    norm_stats = data['norm_stats']
    main_val_dataset = data['main_val_dataset']

    print(f"[INFO] 训练样本: {len(data['main_train_dataset'])}, "
          f"验证样本: {len(data['main_val_dataset'])}")

    print("\n[INFO] 初始化模型...")

    endpoint_predictor = EndpointPredictor(norm_stats=norm_stats).to(device)
    if ENDPOINT_MODEL_PATH.exists():
        ep_ckpt = torch.load(ENDPOINT_MODEL_PATH, map_location=device)
        endpoint_predictor.load_state_dict(ep_ckpt)
        print(f"  EndpointPredictor: 加载自 {ENDPOINT_MODEL_PATH}")
    else:
        print(f"  [WARN] EndpointPredictor 权重文件不存在: {ENDPOINT_MODEL_PATH}")
    endpoint_predictor.eval()
    for p in endpoint_predictor.parameters():
        p.requires_grad = False

    att_module = AttentionModule().to(device)
    att_params = sum(p.numel() for p in att_module.parameters())
    print(f"  AttentionModule: {att_params:,} 参数")

    reward_net = RewardNetwork().to(device)
    rn_params = sum(p.numel() for p in reward_net.parameters())
    print(f"  RewardNetwork: {rn_params:,} 参数")

    ppo_agent = PPOAgent().to(device)
    ppo_params = sum(p.numel() for p in ppo_agent.parameters())
    print(f"  PPOAgent: {ppo_params:,} 参数")

    irl_optimizer = torch.optim.Adam(
        list(att_module.parameters()) + list(reward_net.parameters()),
        lr=REWARD_LR
    )
    ppo_optimizer = torch.optim.Adam(
        ppo_agent.parameters(),
        lr=PPO_LR
    )

    irl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        irl_optimizer, T_max=TOTAL_ALTER_EPOCHS
    )
    ppo_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ppo_optimizer, T_max=TOTAL_ALTER_EPOCHS
    )

    os.makedirs(ATT_DIRL_CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    start_epoch = 0
    phase = 'warmup'
    best_val_loss = float('inf')
    train_log = []

    latest_ckpt = find_latest_checkpoint(ATT_DIRL_CKPT_DIR)
    if latest_ckpt:
        start_epoch, phase, best_val_loss, train_log = load_checkpoint(
            latest_ckpt, att_module, reward_net, ppo_agent,
            irl_optimizer, ppo_optimizer, irl_scheduler, ppo_scheduler
        )
        start_epoch += 1

    if phase == 'warmup' and start_epoch < IRL_WARMUP_EPOCHS:
        print(f"\n{'='*60}")
        print(f"  IRL 热身阶段: epoch {start_epoch} → {IRL_WARMUP_EPOCHS-1}")
        print(f"{'='*60}")

        ppo_agent.eval()

        for epoch in range(start_epoch, IRL_WARMUP_EPOCHS):
            att_module.train()
            reward_net.train()
            epoch_losses = []

            for batch_idx, batch in enumerate(main_train_loader):
                batch = batch_to_device(batch, device)
                B = batch['obs'].shape[0]

                acc_t25 = ensure_tensor(batch['acc_t25'], device)
                ang_t25 = ensure_tensor(batch['ang_t25'], device)

                phi_c = att_module(batch['obs'])

                phi_scalars_expert = reward_net.compute_phi_scalars(
                    pred_action=batch['label_delta'],
                    expert_delta=batch['label_delta'],
                    pos_t25=batch['pos_t25'],
                    acc_t25=acc_t25,
                    ang_t25=ang_t25,
                    ref_frames=batch['ref_frames'],
                )
                R_expert = reward_net(phi_scalars_expert, phi_c, batch['label_delta'])

                log_Z = compute_log_Z_uniform(phi_c, batch, reward_net)

                loss_irl = reward_net.irl_loss(
                    R_expert.squeeze(), log_Z, reward_net.parameters()
                )

                irl_optimizer.zero_grad()
                loss_irl.backward()
                clip_grad_norm_(
                    list(att_module.parameters()) + list(reward_net.parameters()),
                    max_norm=1.0
                )
                irl_optimizer.step()

                epoch_losses.append(loss_irl.item())

            avg_loss = np.mean(epoch_losses)
            train_log.append({
                'epoch': epoch, 'phase': 'warmup', 'irl_loss': avg_loss
            })
            print(f"  [Warmup] Epoch {epoch}/{IRL_WARMUP_EPOCHS-1} | "
                  f"IRL Loss: {avg_loss:.4f} | "
                  f"LR: {irl_optimizer.param_groups[0]['lr']:.2e}")

        phase = 'alternating'
        start_epoch = 0

    elif phase == 'warmup':
        phase = 'alternating'
        start_epoch = 0

    print(f"\n{'='*60}")
    print(f"  交替训练阶段: epoch {start_epoch} → {TOTAL_ALTER_EPOCHS-1}")
    print(f"  IRL步/轮={IRL_STEPS_PER_ALTER}, PPO步/轮={PPO_STEPS_PER_ALTER}")
    print(f"{'='*60}")

    for alter_epoch in range(start_epoch, TOTAL_ALTER_EPOCHS):
        t_start = time.time()
        irl_losses_epoch = []
        ppo_losses_epoch = []

        att_module.train()
        reward_net.train()

        for irl_step in range(IRL_STEPS_PER_ALTER):
            for batch in main_train_loader:
                batch = batch_to_device(batch, device)
                acc_t25 = ensure_tensor(batch['acc_t25'], device)
                ang_t25 = ensure_tensor(batch['ang_t25'], device)

                phi_c = att_module(batch['obs'])

                phi_scalars_expert = reward_net.compute_phi_scalars(
                    pred_action=batch['label_delta'],
                    expert_delta=batch['label_delta'],
                    pos_t25=batch['pos_t25'],
                    acc_t25=acc_t25, ang_t25=ang_t25,
                    ref_frames=batch['ref_frames'],
                )
                R_expert = reward_net(phi_scalars_expert, phi_c, batch['label_delta'])
                log_Z = compute_log_Z_uniform(phi_c, batch, reward_net)
                loss_irl = reward_net.irl_loss(
                    R_expert.squeeze(), log_Z, reward_net.parameters()
                )

                irl_optimizer.zero_grad()
                loss_irl.backward()
                clip_grad_norm_(
                    list(att_module.parameters()) + list(reward_net.parameters()),
                    max_norm=1.0
                )
                irl_optimizer.step()
                irl_losses_epoch.append(loss_irl.item())

        irl_scheduler.step()

        att_module.eval()
        reward_net.eval()
        ppo_agent.train()

        for ppo_step in range(PPO_STEPS_PER_ALTER):
            for batch in main_train_loader:
                batch = batch_to_device(batch, device)
                acc_t25 = ensure_tensor(batch['acc_t25'], device)
                ang_t25 = ensure_tensor(batch['ang_t25'], device)

                with torch.no_grad():
                    phi_c_detach = att_module(batch['obs'])

                ep_info_raw = batch['ep_label_gt'][:, :4]
                ep_info_norm = normalize_ep_info(ep_info_raw, norm_stats)
                actor_input = torch.cat([phi_c_detach, ep_info_norm], dim=1)

                with torch.no_grad():
                    action_old, log_prob_old, _ = ppo_agent.get_action_train(actor_input)

                    phi_scalars = reward_net.compute_phi_scalars(
                        pred_action=action_old,
                        expert_delta=batch['label_delta'],
                        pos_t25=batch['pos_t25'],
                        acc_t25=acc_t25, ang_t25=ang_t25,
                        ref_frames=batch['ref_frames'],
                    )
                    R = reward_net(phi_scalars, phi_c_detach, action_old)

                for _ in range(PPO_UPDATE_EPOCHS):
                    log_prob_new, entropy = ppo_agent.actor.evaluate_action(
                        actor_input, action_old
                    )
                    value = ppo_agent.critic(actor_input)

                    advantage = R.squeeze() - value.squeeze().detach()
                    adv_std = advantage.std() + 1e-8
                    advantage = (advantage - advantage.mean()) / adv_std

                    loss_dict = ppo_agent.compute_ppo_loss(
                        log_prob_new, log_prob_old, advantage,
                        value, R, entropy
                    )

                    ppo_optimizer.zero_grad()
                    loss_dict['loss_ppo'].backward()
                    clip_grad_norm_(ppo_agent.parameters(), max_norm=PPO_GRAD_CLIP)
                    ppo_optimizer.step()

                    ppo_losses_epoch.append(loss_dict['loss_ppo'].item())

        ppo_scheduler.step()

        elapsed = time.time() - t_start
        avg_irl = np.mean(irl_losses_epoch) if irl_losses_epoch else 0
        avg_ppo = np.mean(ppo_losses_epoch) if ppo_losses_epoch else 0

        train_log.append({
            'epoch': alter_epoch,
            'phase': 'alternating',
            'irl_loss': avg_irl,
            'ppo_loss': avg_ppo,
            'irl_lr': irl_optimizer.param_groups[0]['lr'],
            'ppo_lr': ppo_optimizer.param_groups[0]['lr'],
        })

        print(f"  [Alter] Epoch {alter_epoch}/{TOTAL_ALTER_EPOCHS-1} | "
              f"IRL: {avg_irl:.4f} | PPO: {avg_ppo:.4f} | "
              f"Time: {elapsed:.1f}s | "
              f"LR(irl/ppo): {irl_optimizer.param_groups[0]['lr']:.2e}/"
              f"{ppo_optimizer.param_groups[0]['lr']:.2e}")

        if (alter_epoch + 1) % EVAL_INTERVAL == 0:
            print(f"\n  [EVAL] 开始单步评估...")
            val_metrics = evaluate_single_step(
                main_val_loader, att_module, reward_net, ppo_agent, norm_stats, device
            )
            print_metrics_table(val_metrics)

            if val_metrics['irl_loss'] < best_val_loss:
                best_val_loss = val_metrics['irl_loss']
                save_checkpoint(
                    str(ATT_DIRL_MODEL_PATH), alter_epoch, 'alternating',
                    att_module, reward_net, ppo_agent,
                    irl_optimizer, ppo_optimizer, irl_scheduler, ppo_scheduler,
                    best_val_loss, train_log
                )
                print(f"  [BEST] 新最优: IRL Loss = {best_val_loss:.4f}")

        if (alter_epoch + 1) % CKPT_SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(ATT_DIRL_CKPT_DIR, f'ckpt_epoch_{alter_epoch+1}.pth')
            save_checkpoint(
                ckpt_path, alter_epoch, 'alternating',
                att_module, reward_net, ppo_agent,
                irl_optimizer, ppo_optimizer, irl_scheduler, ppo_scheduler,
                best_val_loss, train_log
            )
            cleanup_old_checkpoints(ATT_DIRL_CKPT_DIR, CKPT_KEEP_LAST_N)

    print(f"\n{'='*60}")
    print("  训练完成！")
    print(f"  最优验证IRL Loss: {best_val_loss:.4f}")
    print(f"  权重保存: {ATT_DIRL_MODEL_PATH}")
    print(f"{'='*60}")

    print("\n[INFO] 最终评估...")
    val_metrics = evaluate_single_step(
        main_val_loader, att_module, reward_net, ppo_agent, norm_stats, device
    )
    rollout_metrics = evaluate_rollout(
        main_val_dataset, att_module, ppo_agent, endpoint_predictor,
        norm_stats, device
    )

    print("\n[INFO] V2 滚动评估 (GT-obs)...")
    gt_obs_metrics = evaluate_rollout_v2(
        main_val_dataset, att_module, ppo_agent, endpoint_predictor,
        norm_stats, device, n_trajs=ROLLOUT_EVAL_N, mode='gt_obs'
    )
    print("[INFO] V2 滚动评估 (Autoregressive)...")
    auto_metrics = evaluate_rollout_v2(
        main_val_dataset, att_module, ppo_agent, endpoint_predictor,
        norm_stats, device, n_trajs=ROLLOUT_EVAL_N, mode='autoregressive'
    )

    print_metrics_table(val_metrics, rollout_metrics, gt_obs_metrics, auto_metrics)

    log_path = os.path.join(LOG_DIR, 'train_log.json')
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f"[INFO] 训练日志保存: {log_path}")

    return {
        'att_module': att_module,
        'reward_net': reward_net,
        'ppo_agent': ppo_agent,
        'val_metrics': val_metrics,
        'rollout_metrics': rollout_metrics,
    }


if __name__ == '__main__':
    train()
