import torch
import numpy as np

from config import (
    OBS_WINDOW, PREDICT_STEPS, DT, LANE_WIDTH, INPUT_FEATURE_DIM,
    DURATION_CLIP_RANGE, ROLLOUT_EVAL_N,
    denormalize_delta_acc, denormalize_delta_ang,
)
from reference_path import ReferencePath
from attention_reward import compute_log_Z_uniform, RewardNetwork
from utilities import normalize_ep_info, ensure_tensor, batch_to_device


def kinematic_rollout_5steps(pos_t25, delta_acc, delta_ang, acc_t25, ang_t25, ref_frames):
    B = pos_t25.shape[0]
    device = pos_t25.device

    x  = pos_t25[:, 0].clone()
    y  = pos_t25[:, 1].clone()
    xV = pos_t25[:, 2].clone()
    yV = pos_t25[:, 3].clone()

    positions = torch.zeros(B, PREDICT_STEPS, 2, device=device)

    for k in range(PREDICT_STEPS):
        frac = (k + 1) / PREDICT_STEPS
        acc_new = acc_t25 + frac * delta_acc
        ang_new = ang_t25 + frac * delta_ang

        xV_new = xV + acc_new * torch.cos(ang_new) * DT
        yV_new = yV + acc_new * torch.sin(ang_new) * DT
        x_new = x + xV_new * DT
        y_new = y + yV_new * DT

        ref_x_k = ref_frames[:, k, 0]
        ref_y_k = ref_frames[:, k, 1]
        x_c, y_c, xV_c, yV_c = RewardNetwork._batch_clip_to_corridor(
            x_new, y_new, xV_new, yV_new, ref_x_k, ref_y_k
        )

        positions[:, k, 0] = x_c
        positions[:, k, 1] = y_c
        x, y, xV, yV = x_c, y_c, xV_c, yV_c

    return positions


def evaluate_single_step(val_loader, att_module, reward_net, ppo_agent,
                         norm_stats, device):
    att_module.eval()
    reward_net.eval()
    ppo_agent.eval()

    all_speed_bias = []
    all_heading_bias = []
    all_planning_bias = []
    all_irl_loss = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch_to_device(batch, device)
            B = batch['obs'].shape[0]

            phi_c = att_module(batch['obs'])

            ep_info_raw = batch['ep_label_gt'][:, :4]
            ep_info_norm = normalize_ep_info(ep_info_raw, norm_stats)
            actor_input = torch.cat([phi_c, ep_info_norm], dim=1)

            action_norm = ppo_agent.get_action_infer(actor_input)

            delta_acc_pred = denormalize_delta_acc(action_norm[:, 0])
            delta_ang_pred = denormalize_delta_ang(action_norm[:, 1])
            delta_acc_gt = denormalize_delta_acc(batch['label_delta'][:, 0])
            delta_ang_gt = denormalize_delta_ang(batch['label_delta'][:, 1])

            acc_t25 = ensure_tensor(batch['acc_t25'], device)
            ang_t25 = ensure_tensor(batch['ang_t25'], device)
            acc_pred = acc_t25 + delta_acc_pred
            acc_gt   = acc_t25 + delta_acc_gt
            ang_pred = ang_t25 + delta_ang_pred
            ang_gt   = ang_t25 + delta_ang_gt

            xV_pred = batch['pos_t25'][:, 2] + acc_pred * torch.cos(ang_pred) * DT
            xV_gt   = batch['pos_t25'][:, 2] + acc_gt   * torch.cos(ang_gt) * DT
            speed_bias = (xV_pred - xV_gt).abs().mean().item()
            all_speed_bias.append(speed_bias)

            heading_bias = (ang_pred - ang_gt).abs().mean().item() * (180.0 / np.pi)
            all_heading_bias.append(heading_bias)

            positions_pred = kinematic_rollout_5steps(
                batch['pos_t25'], delta_acc_pred, delta_ang_pred,
                acc_t25, ang_t25, batch['ref_frames']
            )
            pos_pred_5 = positions_pred[:, -1, :]
            ref_5 = batch['ref_frames'][:, -1, :]
            planning_bias = torch.sqrt(
                ((pos_pred_5 - ref_5) ** 2).sum(dim=-1)
            ).mean().item()
            all_planning_bias.append(planning_bias)

            phi_scalars_expert = reward_net.compute_phi_scalars(
                batch['label_delta'], batch['label_delta'],
                batch['pos_t25'], acc_t25, ang_t25, batch['ref_frames']
            )
            R_expert = reward_net(phi_scalars_expert, phi_c, batch['label_delta'])
            log_Z = compute_log_Z_uniform(phi_c, batch, reward_net)
            loss_val = -(R_expert.squeeze() - log_Z).mean().item()
            all_irl_loss.append(loss_val)

    metrics = {
        'speed_bias_ms':    np.mean(all_speed_bias),
        'heading_bias_deg': np.mean(all_heading_bias),
        'planning_bias_m':  np.mean(all_planning_bias),
        'irl_loss':         np.mean(all_irl_loss),
    }
    return metrics


def evaluate_rollout(val_dataset, att_module, ppo_agent, endpoint_predictor,
                     norm_stats, device, n_trajs=ROLLOUT_EVAL_N):
    att_module.eval()
    ppo_agent.eval()
    endpoint_predictor.eval()

    traj_starts = []
    seen_keys = set()
    for i in range(len(val_dataset)):
        s = val_dataset.samples[i]
        if s['window_offset'] == 0:
            key = (round(s['pos_t25'][0], 4), round(s['pos_t25'][1], 4))
            if key not in seen_keys:
                seen_keys.add(key)
                traj_starts.append(s)

    rng = np.random.RandomState(42)
    if len(traj_starts) > n_trajs:
        indices = rng.choice(len(traj_starts), n_trajs, replace=False)
        traj_starts = [traj_starts[i] for i in indices]
    elif len(traj_starts) == 0:
        print("[WARN] 验证集中无可用滚动评估轨迹")
        return {'ADE_m': float('nan'), 'FDE_m': float('nan'),
                'y_final_mean_m': float('nan'), 'duration_mean_s': float('nan'),
                'success_rate': 0.0}

    results = []
    for s in traj_starts:
        try:
            result = _rollout_single_trajectory(
                s, att_module, ppo_agent, endpoint_predictor, norm_stats, device
            )
            results.append(result)
        except Exception as e:
            print(f"  [WARN] 滚动生成失败: {e}")
            continue

    if not results:
        return {'ADE_m': float('nan'), 'FDE_m': float('nan'),
                'y_final_mean_m': float('nan'), 'duration_mean_s': float('nan'),
                'success_rate': 0.0}

    metrics = {
        'ADE_m':          np.mean([r['ade'] for r in results]),
        'FDE_m':          np.mean([r['fde'] for r in results]),
        'y_final_mean_m': np.mean([r['y_final'] for r in results]),
        'duration_mean_s':np.mean([r['duration_s'] for r in results]),
        'success_rate':   np.mean([r['success'] for r in results]),
    }
    return metrics


def _rollout_single_trajectory(sample, att_module, ppo_agent, endpoint_predictor,
                               norm_stats, device):
    with torch.no_grad():
        ep_input = torch.tensor(
            sample['traj_ep_input'], dtype=torch.float32, device=device
        ).unsqueeze(0)
        ep_info_pred = endpoint_predictor.predict(ep_input)

        dx_end = ep_info_pred['delta_xy'][0].item()
        dy_end = ep_info_pred['delta_xy'][1].item()
        xV_end = ep_info_pred['vel_end'][0].item()
        yV_end = ep_info_pred['vel_end'][1].item()
        dur_pred = ep_info_pred['duration'].item()

        ep_info_for_ref = {
            'delta_xy': (dx_end, dy_end),
            'vel_end':  (xV_end, yV_end),
            'duration': dur_pred,
        }

        start_state = sample['start_state']
        ref_traj = ReferencePath.generate(start_state, ep_info_for_ref)
        max_frames = ref_traj.shape[0]
        duration_frames = ref_traj.shape[0]

        ep_info_tensor = torch.tensor(
            [dx_end, dy_end, xV_end, yV_end],
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        ep_info_norm = normalize_ep_info(ep_info_tensor, norm_stats)

        generated_positions = []
        x0 = float(start_state[0])
        y0 = float(start_state[1])
        generated_positions.append((x0, y0))

        obs_tensor = torch.tensor(
            sample['obs_window'], dtype=torch.float32, device=device
        ).unsqueeze(0)

        cur_x  = float(sample['pos_t25'][0])
        cur_y  = float(sample['pos_t25'][1])
        cur_xV = float(sample['pos_t25'][2])
        cur_yV = float(sample['pos_t25'][3])
        cur_acc = float(sample['acc_t25'])
        cur_ang = float(sample['ang_t25'])

        frame_count = OBS_WINDOW
        step_count = 0
        max_steps = (max_frames - OBS_WINDOW) // PREDICT_STEPS

        while step_count < max_steps and frame_count < max_frames:
            phi_c = att_module(obs_tensor)
            actor_input = torch.cat([phi_c, ep_info_norm], dim=1)
            action_norm = ppo_agent.get_action_infer(actor_input)

            delta_acc = denormalize_delta_acc(action_norm[0, 0]).item()
            delta_ang = denormalize_delta_ang(action_norm[0, 1]).item()

            for k in range(PREDICT_STEPS):
                frac = (k + 1) / PREDICT_STEPS
                acc_new = cur_acc + frac * delta_acc
                ang_new = cur_ang + frac * delta_ang

                xV_new = cur_xV + acc_new * np.cos(ang_new) * DT
                yV_new = cur_yV + acc_new * np.sin(ang_new) * DT
                x_new = cur_x + xV_new * DT
                y_new = cur_y + yV_new * DT

                ref_idx = frame_count + k
                if ref_idx < len(ref_traj):
                    ref_x_k = ref_traj[ref_idx, 0]
                    ref_y_k = ref_traj[ref_idx, 1]
                else:
                    ref_x_k = ref_traj[-1, 0]
                    ref_y_k = ref_traj[-1, 1]

                x_c, y_c, xV_c, yV_c = ReferencePath.clip_to_corridor(
                    x_new, y_new, xV_new, yV_new, ref_x_k, ref_y_k
                )

                cur_x, cur_y, cur_xV, cur_yV = x_c, y_c, xV_c, yV_c
                generated_positions.append((cur_x, cur_y))

            cur_acc = cur_acc + delta_acc
            cur_ang = cur_ang + delta_ang
            frame_count += PREDICT_STEPS
            step_count += 1

            y_displacement = abs(cur_y - y0)
            if y_displacement >= LANE_WIDTH * 0.95:
                break

        gen_traj = np.array(generated_positions)
        total_gen = len(gen_traj)

        gt_len = min(total_gen, len(ref_traj))
        gen_subset = gen_traj[:gt_len]
        ref_subset = ref_traj[:gt_len]

        ade = np.sqrt(((gen_subset - ref_subset) ** 2).sum(axis=-1)).mean()
        fde = np.sqrt(((gen_subset[-1] - ref_subset[-1]) ** 2).sum())
        y_final = gen_traj[-1, 1]
        duration_s = (total_gen - 1) * DT
        y_disp = y_final - y0
        success = (0.0 <= abs(y_disp) <= 4.0)

    return {
        'ade': float(ade),
        'fde': float(fde),
        'y_final': float(y_final),
        'duration_s': float(duration_s),
        'success': float(success),
    }


def _build_obs_from_sample(sample, norm_stats):
    window_norm = sample['window_raw_norm']
    pos_window = sample['position_window']

    rel_x = pos_window[:, 0] - pos_window[0, 0]
    rel_y = pos_window[:, 1] - pos_window[0, 1]
    rel_x_norm = (rel_x - norm_stats['EGO_REL_X']['mean']) / norm_stats['EGO_REL_X']['std']
    rel_y_norm = (rel_y - norm_stats['EGO_REL_Y']['mean']) / norm_stats['EGO_REL_Y']['std']

    obs = np.zeros((OBS_WINDOW, INPUT_FEATURE_DIM), dtype=np.float32)
    obs[:, 0] = window_norm[:, 0]
    obs[:, 1] = window_norm[:, 1]
    obs[:, 2] = window_norm[:, 2]
    obs[:, 3] = window_norm[:, 3]
    obs[:, 4] = rel_x_norm.astype(np.float32)
    obs[:, 5] = rel_y_norm.astype(np.float32)
    obs[:, 6:22] = window_norm[:, 4:20]

    return obs


def _group_trajectory_windows(val_dataset):
    traj_map = {}
    for i in range(len(val_dataset)):
        s = val_dataset.samples[i]
        key = (round(float(s['start_state'][0]), 4),
               round(float(s['start_state'][1]), 4))
        traj_map.setdefault(key, []).append(s)

    result = []
    for key, windows in traj_map.items():
        windows.sort(key=lambda s: s['window_offset'])
        result.append(windows)
    return result


def _reconstruct_gt_trajectory(windows):
    full_pos = windows[0]['position_window'].copy()
    for w in windows[1:]:
        new_frames = w['position_window'][-PREDICT_STEPS:]
        full_pos = np.vstack([full_pos, new_frames])
    return full_pos


def _reconstruct_gt_ego_states(windows):
    acc_list = []
    ang_list = []
    pos_list = []
    for w in windows:
        acc_list.append(float(w['acc_t25']))
        ang_list.append(float(w['ang_t25']))
        pos_list.append(w['pos_t25'].copy())
    return {'acc_at_t25': acc_list, 'ang_at_t25': ang_list, 'pos_t25': pos_list}


def _build_obs_from_history(all_x, all_y, all_xV, all_yV, all_acc,
                            all_heading, surr_features, win_start, norm_stats):
    win_end = win_start + OBS_WINDOW

    w_x  = np.array(all_x[win_start:win_end], dtype=np.float64)
    w_y  = np.array(all_y[win_start:win_end], dtype=np.float64)
    w_xV = np.array(all_xV[win_start:win_end], dtype=np.float64)
    w_yV = np.array(all_yV[win_start:win_end], dtype=np.float64)
    w_acc = np.array(all_acc[win_start:win_end], dtype=np.float64)
    w_heading = np.array(all_heading[win_start:win_end], dtype=np.float64)

    xV_norm = ((w_xV - norm_stats['EGO_XV']['mean']) / norm_stats['EGO_XV']['std']).astype(np.float32)
    yV_norm = ((w_yV - norm_stats['EGO_YV']['mean']) / norm_stats['EGO_YV']['std']).astype(np.float32)
    heading_norm = ((w_heading - norm_stats['EGO_HEADING']['mean']) /
                    norm_stats['EGO_HEADING']['std']).astype(np.float32)
    acc_norm = ((w_acc - norm_stats['EGO_ACC']['mean']) / norm_stats['EGO_ACC']['std']).astype(np.float32)

    rel_x = w_x - w_x[0]
    rel_y = w_y - w_y[0]
    rel_x_norm = ((rel_x - norm_stats['EGO_REL_X']['mean']) /
                  norm_stats['EGO_REL_X']['std']).astype(np.float32)
    rel_y_norm = ((rel_y - norm_stats['EGO_REL_Y']['mean']) /
                  norm_stats['EGO_REL_Y']['std']).astype(np.float32)

    new_surr = np.zeros((OBS_WINDOW, 16), dtype=np.float32)
    new_surr[:OBS_WINDOW - PREDICT_STEPS] = surr_features[PREDICT_STEPS:]
    new_surr[OBS_WINDOW - PREDICT_STEPS:] = surr_features[-1:]
    surr_updated = new_surr

    obs = np.zeros((OBS_WINDOW, INPUT_FEATURE_DIM), dtype=np.float32)
    obs[:, 0] = xV_norm
    obs[:, 1] = yV_norm
    obs[:, 2] = heading_norm
    obs[:, 3] = acc_norm
    obs[:, 4] = rel_x_norm
    obs[:, 5] = rel_y_norm
    obs[:, 6:22] = surr_updated

    return obs, surr_updated


def evaluate_rollout_v2(val_dataset, att_module, ppo_agent, ep_model,
                        norm_stats, device, n_trajs=50, mode='gt_obs'):
    att_module.eval()
    ppo_agent.eval()
    ep_model.eval()

    traj_groups = _group_trajectory_windows(val_dataset)
    valid_groups = [g for g in traj_groups if len(g) >= 2]

    rng = np.random.RandomState(42)
    if len(valid_groups) > n_trajs:
        indices = rng.choice(len(valid_groups), n_trajs, replace=False)
        valid_groups = [valid_groups[i] for i in indices]

    print(f"  [V2-{mode}] 评估轨迹数: {len(valid_groups)}")

    results = []
    for i, windows in enumerate(valid_groups):
        try:
            gt_traj = _reconstruct_gt_trajectory(windows)
            if mode == 'gt_obs':
                r = _rollout_gt_obs_single(
                    windows, att_module, ppo_agent, ep_model,
                    norm_stats, device, gt_traj
                )
            elif mode == 'autoregressive':
                r = _rollout_autoregressive_single(
                    windows, att_module, ppo_agent, ep_model,
                    norm_stats, device, gt_traj
                )
            else:
                raise ValueError(f"未知模式: {mode}")
            r['sample_idx'] = i
            results.append(r)
        except Exception as e:
            print(f"    [WARN] 轨迹 {i} 失败: {e}")
            continue

    if not results:
        return {'ADE_m': float('nan'), 'FDE_m': float('nan'),
                'y_disp_mean': float('nan'), 'success_rate': 0.0,
                'results': []}

    metrics = {
        'ADE_m':       np.mean([r['ade'] for r in results]),
        'FDE_m':       np.mean([r['fde'] for r in results]),
        'y_disp_mean': np.mean([r['y_disp'] for r in results]),
        'success_rate': np.mean([float(r['success']) for r in results]),
        'results':     results,
    }

    print(f"    ADE={metrics['ADE_m']:.4f}m  FDE={metrics['FDE_m']:.4f}m  "
          f"Success={metrics['success_rate']:.2%}  y_disp={metrics['y_disp_mean']:.3f}m")

    return metrics


def print_metrics_table(single_step_metrics, rollout_metrics=None,
                       gt_obs_metrics=None, auto_metrics=None):
    print("\n" + "=" * 50)
    print("   单步评估指标 (Table 3)")
    print("=" * 50)
    if single_step_metrics:
        print(f"  Speed Bias:    {single_step_metrics.get('speed_bias_ms', 0):.4f} m/s")
        print(f"  Heading Bias:  {single_step_metrics.get('heading_bias_deg', 0):.4f} °")
        print(f"  Planning Bias: {single_step_metrics.get('planning_bias_m', 0):.4f} m")
        print(f"  Val IRL Loss:  {single_step_metrics.get('irl_loss', 0):.4f}")

    if rollout_metrics is not None:
        print("\n" + "=" * 50)
        print("   滚动生成评估指标 (旧版 obs-reuse)")
        print("=" * 50)
        print(f"  ADE:           {rollout_metrics['ADE_m']:.4f} m")
        print(f"  FDE:           {rollout_metrics['FDE_m']:.4f} m")
        print(f"  Y Final Mean:  {rollout_metrics.get('y_final_mean_m', 0):.4f} m")
        print(f"  Duration Mean: {rollout_metrics.get('duration_mean_s', 0):.4f} s")
        print(f"  Success Rate:  {rollout_metrics['success_rate']:.2%}")

    if gt_obs_metrics is not None:
        print("\n" + "=" * 50)
        print("   V2 滚动评估 — GT-obs (教师强迫)")
        print("=" * 50)
        print(f"  ADE:           {gt_obs_metrics['ADE_m']:.4f} m")
        print(f"  FDE:           {gt_obs_metrics['FDE_m']:.4f} m")
        print(f"  Y Disp Mean:   {gt_obs_metrics['y_disp_mean']:.4f} m")
        print(f"  Success Rate:  {gt_obs_metrics['success_rate']:.2%}")

    if auto_metrics is not None:
        print("\n" + "=" * 50)
        print("   V2 滚动评估 — Autoregressive (自回归)")
        print("=" * 50)
        print(f"  ADE:           {auto_metrics['ADE_m']:.4f} m")
        print(f"  FDE:           {auto_metrics['FDE_m']:.4f} m")
        print(f"  Y Disp Mean:   {auto_metrics['y_disp_mean']:.4f} m")
        print(f"  Success Rate:  {auto_metrics['success_rate']:.2%}")

    print("=" * 50)
