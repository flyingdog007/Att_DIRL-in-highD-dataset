import os
import glob
import torch
import numpy as np

from config import (
    ENDPOINT_OUTPUT,
    CKPT_KEEP_LAST_N,
)


def normalize_ep_info(ep_info_raw, norm_stats):
    device = ep_info_raw.device
    mean = torch.tensor(
        norm_stats['ENDPOINT_OUTPUT']['mean'][:4],
        dtype=torch.float32, device=device
    )
    std = torch.tensor(
        norm_stats['ENDPOINT_OUTPUT']['std'][:4],
        dtype=torch.float32, device=device
    )
    return (ep_info_raw - mean) / (std + 1e-8)


def ensure_tensor(val, device, dtype=torch.float32):
    if isinstance(val, torch.Tensor):
        return val.to(device=device, dtype=dtype)
    elif isinstance(val, (list, np.ndarray)):
        return torch.tensor(val, dtype=dtype, device=device)
    else:
        return torch.tensor([val], dtype=dtype, device=device)


def batch_to_device(batch, device):
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def save_checkpoint(path, epoch, phase, att_module, reward_net, ppo_agent,
                    irl_optimizer, ppo_optimizer, irl_scheduler, ppo_scheduler,
                    best_val_loss, train_log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'att_module_state': att_module.state_dict(),
        'reward_net_state': reward_net.state_dict(),
        'actor_state': ppo_agent.actor.state_dict(),
        'critic_state': ppo_agent.critic.state_dict(),
        'irl_optim_state': irl_optimizer.state_dict(),
        'ppo_optim_state': ppo_optimizer.state_dict(),
        'irl_sched_state': irl_scheduler.state_dict(),
        'ppo_sched_state': ppo_scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_log': train_log,
    }
    torch.save(checkpoint, path)
    print(f"  [CKPT] 保存: {path}")


def load_checkpoint(path, att_module, reward_net, ppo_agent,
                    irl_optimizer, ppo_optimizer, irl_scheduler, ppo_scheduler):
    checkpoint = torch.load(path, map_location='cpu')
    att_module.load_state_dict(checkpoint['att_module_state'])
    reward_net.load_state_dict(checkpoint['reward_net_state'])
    ppo_agent.actor.load_state_dict(checkpoint['actor_state'])
    ppo_agent.critic.load_state_dict(checkpoint['critic_state'])
    irl_optimizer.load_state_dict(checkpoint['irl_optim_state'])
    ppo_optimizer.load_state_dict(checkpoint['ppo_optim_state'])
    irl_scheduler.load_state_dict(checkpoint['irl_sched_state'])
    ppo_scheduler.load_state_dict(checkpoint['ppo_sched_state'])
    print(f"  [CKPT] 恢复: {path}, epoch={checkpoint['epoch']}, phase={checkpoint['phase']}")
    return (checkpoint['epoch'], checkpoint['phase'],
            checkpoint['best_val_loss'], checkpoint['train_log'])


def cleanup_old_checkpoints(ckpt_dir, keep_n=CKPT_KEEP_LAST_N):
    pattern = os.path.join(ckpt_dir, 'ckpt_epoch_*.pth')
    ckpt_files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(ckpt_files) > keep_n:
        for f in ckpt_files[:-keep_n]:
            os.remove(f)
            print(f"  [CKPT] 删除旧checkpoint: {f}")


def find_latest_checkpoint(ckpt_dir):
    pattern = os.path.join(ckpt_dir, 'ckpt_epoch_*.pth')
    ckpt_files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if ckpt_files:
        return ckpt_files[-1]
    return None
