# validate_src_tgt_loss.py
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from tqdm import tqdm

from data import SpectrogramFilterDataset
from model.loss import MultiResolutionSTFTLoss


# ---------------- DDP utils ----------------
def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1

def setup_ddp():
    """
    Initialize Distributed Data Parallel from torchrun (env://).
    """
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = get_rank()
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def ddp_all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """
    Average a scalar tensor across all processes.
    """
    if not is_dist_available_and_initialized():
        return t
    tmp = t.clone()
    dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
    tmp /= get_world_size()
    return tmp


# ---------------- Loader builder ----------------
def build_valid_loader(cfg, rank: int, world_size: int) -> DataLoader:
    """
    Build DDP-valid DataLoader with DistributedSampler.
    """
    # choose per-GPU batch size (fallback to training batch size)
    global_bsz = int(cfg.training.batch_size)
    per_gpu_bsz = max(1, global_bsz // max(1, world_size))

    # pick valid roots; fallback to train roots if not provided
    valid_root_dirs = (
        cfg.dataset.valid_root_dirs
        if "valid_root_dirs" in cfg.dataset
        else cfg.dataset.root_dirs
    )

    valid_dataset = SpectrogramFilterDataset(
        root_dirs=valid_root_dirs,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        window=cfg.dataset.window,
        use_log_mag=cfg.dataset.use_log_mag,
        use_mel=False,
        n_mel_bins=cfg.dataset.n_mel_bins,
        use_precomputed=cfg.dataset.use_precomputed,
        cache_subdir=cfg.dataset.cache_subdir,
    )

    sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    loader = DataLoader(
        valid_dataset,
        batch_size=per_gpu_bsz,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(8 > 0),
    )
    return loader


# ---------------- Eval core ----------------
@torch.no_grad()
def eval_src_vs_tgt_loss(criterion, dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute DDP-averaged Multi-Resolution STFT loss between src_audio and tgt_audio.
    """
    rank = get_rank()
    running = 0.0
    batches = 0

    iterable = dataloader if rank != 0 else tqdm(dataloader, desc="Valid (src↔tgt)")

    for batch in iterable:
        src_audio = batch["src_audio"].to(device, non_blocking=True)
        tgt_audio = batch["tgt_audio"].to(device, non_blocking=True)

        # dataset may return [B, 1, T]; squeeze to [B, T]
        if src_audio.dim() == 3 and src_audio.size(1) == 1:
            src_audio = src_audio.squeeze(1)
        if tgt_audio.dim() == 3 and tgt_audio.size(1) == 1:
            tgt_audio = tgt_audio.squeeze(1)

        # crop to same length
        min_len = min(src_audio.shape[-1], tgt_audio.shape[-1])
        if src_audio.shape[-1] != min_len:
            src_audio = src_audio[..., :min_len]
        if tgt_audio.shape[-1] != min_len:
            tgt_audio = tgt_audio[..., :min_len]

        loss = criterion(src_audio, tgt_audio)  # scalar tensor
        running += float(loss.detach().item())
        batches += 1

    local_avg = torch.tensor(0.0, device=device) if batches == 0 else torch.tensor(running / batches, device=device)
    global_avg = ddp_all_reduce_mean(local_avg)
    return float(global_avg.item())


def main():
    # init DDP
    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    # load config
    cfg = OmegaConf.load("default.yaml")
    if rank == 0:
        print(f"[DDP] world_size={world_size}, rank={rank}, device={device}")

    # loss
    loss_cfg = cfg.loss
    criterion = MultiResolutionSTFTLoss(
        fft_sizes=loss_cfg.fft_sizes,
        hop_sizes=loss_cfg.hop_sizes,
        win_lengths=loss_cfg.win_lengths,
        mag_weight=loss_cfg.mag_weight,
        sc_weight=loss_cfg.sc_weight,
        use_log_mag=loss_cfg.use_log_mag,
    ).to(device)

    # loader
    valid_loader = build_valid_loader(cfg, rank, world_size)

    # run eval
    loss_val = eval_src_vs_tgt_loss(criterion, valid_loader, device)

    if rank == 0:
        print(f"✅ Global Src↔Tgt MR-STFT Loss (valid): {loss_val:.6f}")

    # finalize
    cleanup_ddp()


if __name__ == "__main__":
    # run with: torchrun --standalone --nnodes=1 --nproc_per_node=<NGPUS> validate_src_tgt_loss.py
    main()
