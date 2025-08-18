import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

from data import SpectrogramFilterDataset
# CHANGED: 모델명 교체
from model import AudioFilterModel
from model.loss import MultiResolutionSTFTLoss
from logs.log_utils import *

# ---------------- DDP utils ----------------
def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1

def setup_ddp():
    # env:// expects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE from torchrun
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = get_rank()
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def ddp_all_reduce_mean(tensor: torch.Tensor):
    if not is_dist_available_and_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t

# -------------- (optional) mask util --------------
def make_note_on_mask(src_mag, sample_rate, hop_length):
    """
    src_mag: [B, F, T]
    returns: [B, T] (float 0/1)
    """
    B, _, T = src_mag.shape
    on_frames = min(int(3.0 * sample_rate / hop_length), T)
    mask = torch.zeros(B, T, device=src_mag.device, dtype=src_mag.dtype)
    mask[:, :on_frames] = 1.0
    return mask

# -------------- validate (DDP-aware) --------------
@torch.no_grad()
def validate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_batches = 0

    for batch in dataloader:
        src_audio = batch["src_audio"].to(device, non_blocking=True)
        tgt_audio = batch["tgt_audio"].to(device, non_blocking=True).squeeze(1)
        src_mag   = batch["src_mag"].to(device, non_blocking=True)
        src_phase = batch["src_phase"].to(device, non_blocking=True)
        tgt_mag   = batch["tgt_mag"].to(device, non_blocking=True)

        note_on_mask = batch["note_on_mask"].to(device, non_blocking=True)
        if note_on_mask.dim() == 3 and note_on_mask.size(1) == 1:
            note_on_mask = note_on_mask.squeeze(1)  # [B, T]

        # NEW: preset_vec (FiLM cond)
        cond = batch.get("preset_vec", None)
        if cond is not None:
            cond = cond.to(device, non_blocking=True)

        pred_audio = model(
            src_mag=src_mag,
            src_phase=src_phase,
            tgt_mag=tgt_mag,
            src_audio_length=src_audio.shape[-1],
            note_on_mask=note_on_mask,
            cond=cond,                         # <---- 여기
        )

        min_len = min(pred_audio.shape[-1], tgt_audio.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        tgt_audio  = tgt_audio[...,  :min_len]

        loss = criterion(pred_audio, tgt_audio)
        running_loss += loss.detach()
        running_batches += 1

    # 프로세스별 평균 → 전역 평균
    if running_batches == 0:
        local_avg = torch.tensor(0.0, device=device)
    else:
        local_avg = running_loss / running_batches  # scalar tensor

    global_avg = ddp_all_reduce_mean(local_avg)
    return float(global_avg.item())

# -------------- train (DDP) --------------
def train():
    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    cfg = OmegaConf.load("default.yaml")
    if rank == 0:
        print(f"Using DDP on {world_size} GPUs. Rank {rank} on device {device}.")

    best_val_loss = float("inf")
    best_epoch = -1

    # rank0만 로거/디렉토리 초기화
    if rank == 0:
        init_loggers(cfg.log)
        os.makedirs(cfg.log.ckpt_dir, exist_ok=True)

    # ===== Dataset / Sampler / Loader =====
    per_gpu_batch = max(1, cfg.training.batch_size // world_size)

    train_dataset = SpectrogramFilterDataset(
        root_dirs=cfg.dataset.root_dirs,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        window=cfg.dataset.window,
        use_log_mag=cfg.dataset.use_log_mag,
        use_mel=False,
        n_mel_bins=cfg.dataset.n_mel_bins,
        use_precomputed=cfg.dataset.use_precomputed,
        cache_subdir=cfg.dataset.cache_subdir
        # NOTE: preset_vec은 데이터셋이 parent에서 자동 로드하도록 구현되어 있다고 가정
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=(8 > 0),
    )

    valid_dataset = SpectrogramFilterDataset(
        root_dirs=cfg.dataset.valid_root_dirs,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        window=cfg.dataset.window,
        use_log_mag=cfg.dataset.use_log_mag,
        use_mel=False,
        n_mel_bins=cfg.dataset.n_mel_bins,
        use_precomputed=cfg.dataset.use_precomputed,
        cache_subdir=cfg.dataset.cache_subdir
    )
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_gpu_batch,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=(8 > 0),
    )

    # ===== Model =====

    model = AudioFilterModel(
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        sample_rate=cfg.dataset.sample_rate,
        embed_dim=cfg.model.embed_dim,
        cnn_layers=cfg.model.cnn_layers,
        attn_heads=cfg.model.attn_heads,
        use_self_attn=cfg.model.use_self_attn,
        use_cross_attn=cfg.model.use_cross_attn,
        n_self_attn=cfg.model.n_self_attn,
        n_cross_attn=cfg.model.n_cross_attn,
        filter_activation=cfg.filter.activation,
        enable_inject=cfg.model.enable_inject,
        inject_f0_hz=cfg.model.inject_f0_hz,
        inject_sigma_hz=cfg.model.inject_sigma_hz,
        smooth_gate_kernel=cfg.model.smooth_gate_kernel,
        # ----- FiLM -----
        cond_dim=cfg.model.cond_dim,
        use_film=cfg.model.use_film,
        film_on_tgt=cfg.model.film_on_tgt,
        film_hidden=cfg.model.film_hidden,
        film_dropout=cfg.model.film_dropout,
        film_layernorm=cfg.model.film_layernorm
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # ===== Loss / Optim / Sched =====
    loss_cfg = cfg.loss
    criterion = MultiResolutionSTFTLoss(
        fft_sizes=loss_cfg.fft_sizes,
        hop_sizes=loss_cfg.hop_sizes,
        win_lengths=loss_cfg.win_lengths,
        mag_weight=loss_cfg.mag_weight,
        sc_weight=loss_cfg.sc_weight,
        use_log_mag=loss_cfg.use_log_mag
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)

    # ===== Logging / Resume =====
    writer = SummaryWriter(log_dir=cfg.log.ckpt_dir) if rank == 0 else None

    start_epoch = 0
    step = 0

    if cfg.training.resume and rank == 0:
        # rank0에서만 로드 후, 파라미터 브로드캐스트는 DDP가 자동 동기화
        model, start_epoch = load_checkpoint(model, start_epoch, cfg.log, map_location=device, tag=cfg.training.resume_tag)
        print(f"▶ [rank0] Resumed from epoch {start_epoch}")

        log_path = os.path.join(cfg.log.ckpt_dir, "loss_log.csv")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    try:
                        step = int(last_line.strip().split(",")[0]) + 1
                    except Exception:
                        print("⚠️ Could not parse step from CSV. Starting from step=0")
        print(f"▶ [rank0] Starting from step {step}")

    # 모든 rank가 resume 동기화 지점까지 대기
    if is_dist_available_and_initialized():
        dist.barrier()

    # ===== Train Loop =====
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # sampler에 epoch 제공(셔플 바뀌게)
        train_sampler.set_epoch(epoch)
        if valid_sampler is not None and hasattr(valid_sampler, "set_epoch"):
            valid_sampler.set_epoch(epoch)

        model.train()
        running = 0.0

        # tqdm은 rank0만
        iterable = train_loader if rank != 0 else tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

        for batch in iterable:
            src_audio = batch["src_audio"].to(device, non_blocking=True)
            tgt_audio = batch["tgt_audio"].to(device, non_blocking=True).squeeze(1)
            src_mag   = batch["src_mag"].to(device, non_blocking=True)
            src_phase = batch["src_phase"].to(device, non_blocking=True)
            tgt_mag   = batch["tgt_mag"].to(device, non_blocking=True)

            note_on_mask = batch["note_on_mask"].to(device, non_blocking=True)
            if note_on_mask.dim() == 3 and note_on_mask.size(1) == 1:
                note_on_mask = note_on_mask.squeeze(1)  # [B, T]

            # NEW: preset_vec
            cond = batch.get("preset_vec", None)
            if cond is not None:
                cond = cond.to(device, non_blocking=True)

            pred_audio = model(
                src_mag=src_mag,
                src_phase=src_phase,
                tgt_mag=tgt_mag,
                src_audio_length=src_audio.shape[-1],
                note_on_mask=note_on_mask,
                cond=cond,                      # <---- 여기
            )

            min_len = min(pred_audio.shape[-1], tgt_audio.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            tgt_audio  = tgt_audio[...,  :min_len]

            loss = criterion(pred_audio, tgt_audio)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            running += loss.detach().item()

            # rank0만 CSV/TB 로깅
            if rank == 0:
                log_metrics(step, epoch+1, float(loss.detach().item()))
                step += 1

        # 에폭 평균(로컬) → 전역 평균으로 집계해서 출력/로깅
        local_avg = torch.tensor(running / max(1, len(train_loader)), device=device)
        global_train_avg = ddp_all_reduce_mean(local_avg).item()

        if rank == 0:
            log_epoch(epoch+1, global_train_avg)
            writer.add_scalar("train/avg_loss", global_train_avg, epoch+1)
            print(f"[Epoch {epoch+1}] Train Avg Loss: {global_train_avg:.4f}")

        # === Validation ===
        val_loss = validate(model, criterion, valid_loader, device)
        if rank == 0:
            writer.add_scalar("valid/loss", val_loss, epoch+1)
            print(f"[Epoch {epoch+1}] Valid Loss: {val_loss:.4f}")

        # Save latest/best on rank0
        if rank == 0:
            # save latest
            save_checkpoint(model, epoch+1, cfg.log)

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                save_checkpoint(model, best_epoch, cfg.log, tag="best")
                print(f"💾 Best model updated at epoch {best_epoch} (val_loss: {val_loss:.6f})")

        scheduler.step()

        # 모든 rank 동기화 (선택)
        if is_dist_available_and_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    if rank == 0:
        print("✅ Training complete.")
    cleanup_ddp()

if __name__ == "__main__":
    # torchrun이 env 세팅함
    train()

# torchrun --nproc_per_node=2 train.py