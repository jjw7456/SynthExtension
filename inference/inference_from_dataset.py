# inference_valid_ddp_samples.py
import os, time, random
import torch
import soundfile as sf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from collections import defaultdict

from data import SpectrogramFilterDataset
from model import AudioFilterModel
from logs.log_utils import load_checkpoint


# ---------------- DDP utils ----------------
def ddp_env_present():
    return all(k in os.environ for k in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"])

def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def setup_ddp(backend="nccl", device_type="cuda"):
    if ddp_env_present():
        dist.init_process_group(backend=backend, init_method="env://")
        if device_type == "cuda":
            torch.cuda.set_device(get_rank())

def cleanup_ddp():
    if is_dist_ready():
        dist.barrier()
        dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if is_dist_ready() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_ready() else 1

def allreduce_sum(value: float) -> float:
    if not is_dist_ready():
        return value
    t = torch.tensor([value], device=f"cuda:{get_rank()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


# ---------------- I/O helpers ----------------
def save_wav(path, tensor_1d, sr):
    x = tensor_1d.detach().cpu().float().numpy()
    sf.write(path, x, samplerate=int(sr))

def pack_alpha2_tensors(core, device):
    sums, cnts, tags = [], [], []
    # self
    if hasattr(core.encoder, "self_attn_blocks"):
        for i, blk in enumerate(core.encoder.self_attn_blocks):
            if hasattr(blk, "alpha2_sum"):
                sums.append(blk.alpha2_sum.to(device))
                cnts.append(blk.alpha2_cnt.to(device))
                tags.append(f"self[{i}]")
    # cross
    if hasattr(core, "cross_attn_blocks"):
        for i, blk in enumerate(core.cross_attn_blocks):
            if hasattr(blk, "alpha2_sum"):
                sums.append(blk.alpha2_sum.to(device))
                cnts.append(blk.alpha2_cnt.to(device))
                tags.append(f"cross[{i}]")
    if not sums:
        return None, None, []
    return torch.stack(sums), torch.stack(cnts), tags

# ---------------- main inference loop ----------------
@torch.no_grad()
def run_inference_on_indices(model, dataset, indices, device, outdir, sr, cond_dim=None):
    """
    model: DDP-wrapped or plain model (eval mode)
    dataset: SpectrogramFilterDataset
    indices: list[int] to process on THIS rank
    """
    os.makedirs(outdir, exist_ok=True)
    rank = get_rank()

    local_time_sum = 0.0
    local_count = 0

    meter = defaultdict(float); n=0

    for idx in indices:
        sample = dataset[idx]

        # ---- pull tensors and fix dims ----
        src_audio = sample["src_audio"].to(device)
        tgt_audio = sample["tgt_audio"].to(device)
        if src_audio.dim() == 1:  # [T] -> [1,T]
            src_audio = src_audio.unsqueeze(0)
        if tgt_audio.dim() == 1:
            tgt_audio = tgt_audio.unsqueeze(0)

        src_mag   = sample["src_mag"].to(device)     # [F,Tt] or [1,F,Tt]
        src_phase = sample["src_phase"].to(device)
        tgt_mag   = sample["tgt_mag"].to(device)

        if src_mag.dim() == 2:   src_mag   = src_mag.unsqueeze(0)
        if src_phase.dim() == 2: src_phase = src_phase.unsqueeze(0)
        if tgt_mag.dim() == 2:   tgt_mag   = tgt_mag.unsqueeze(0)

        # optional fields
        note_on_mask = sample.get("note_on_mask", None)
        if note_on_mask is None:
            Tt = src_mag.shape[-1]
            note_on_mask = torch.ones(1, Tt, device=device, dtype=src_mag.dtype)
        else:
            note_on_mask = note_on_mask.to(device)
            if note_on_mask.dim() == 1:          # [Tt] -> [1,Tt]
                note_on_mask = note_on_mask.unsqueeze(0)
            elif note_on_mask.dim() == 3 and note_on_mask.size(1) == 1:  # [1,1,Tt] -> [1,Tt]
                note_on_mask = note_on_mask.squeeze(1)

        cond = sample.get("preset_vec", None)
        if cond is not None:
            cond = cond.to(device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)  # [1,D]
            if cond_dim is not None and cond.shape[-1] != cond_dim:
                d = cond.shape[-1]
                if d > cond_dim:
                    cond = cond[..., :cond_dim]
                else:
                    pad = torch.zeros(cond.shape[:-1] + (cond_dim - d,), device=device, dtype=cond.dtype)
                    cond = torch.cat([cond, pad], dim=-1)

        # ---- forward & timing ----
        t0 = time.perf_counter()
        pred_audio, dbg = model(
            src_mag=src_mag,
            src_phase=src_phase,
            tgt_mag=tgt_mag,
            src_audio_length=src_audio.shape[-1],
            note_on_mask=note_on_mask,
            cond=cond,
            tgt_audio=tgt_audio,
            collect_metrics=True
        )  # [1,T]
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) * 1000.0  # ms

        pred_audio = pred_audio.squeeze(0)
        tgt_audio  = tgt_audio.squeeze(0)
        src_audio  = src_audio.squeeze(0)

        # ---- align lengths ----
        min_len = min(pred_audio.shape[-1], tgt_audio.shape[-1], src_audio.shape[-1])
        pred_audio = pred_audio[..., :min_len]
        tgt_audio  = tgt_audio[...,  :min_len]
        src_audio  = src_audio[...,  :min_len]

        # ---- save ----
        base = f"valid_{idx:06d}_r{rank}"
        save_wav(os.path.join(outdir, f"{base}_src.wav"),  src_audio,  sr)
        save_wav(os.path.join(outdir, f"{base}_tgt.wav"),  tgt_audio,  sr)
        save_wav(os.path.join(outdir, f"{base}_pred.wav"), pred_audio, sr)

        print(f"[rank {rank}] idx={idx}  forward_time={dt:.2f} ms  saved→ {base}_*.wav")
        local_time_sum += dt
        local_count += 1

        # 루프 안에서
        for k,v in dbg.items():
            if isinstance(v,(int,float)) and v is not None:
                meter[k]+=float(v)
        n+=1

    # 루프 끝
    print("=== Averages ===")
    for k,v in meter.items():
        print(f"{k}: {v/n:.6f}")

    return local_time_sum, local_count


def main(n_samples=5, epoch=None, tag=None, outdir="inference_valid_samples",
         device_arg="auto", backend="nccl"):
    
    if device_arg == "auto":
        use_cuda = torch.cuda.is_available()
    elif device_arg == "cuda":
        use_cuda = True
    else:  # "cpu"
        use_cuda = False

    # --- DDP init (optional) ---
    setup_ddp(backend=("nccl" if use_cuda else "gloo") if backend=="nccl" else backend,
              device_type=("cuda" if use_cuda else "cpu"))
    rank = get_rank()
    world = get_world_size()
    

    device = torch.device(f"cuda:{get_rank()}" if use_cuda else "cpu")

    # --- cfg / dataset ---
    cfg = OmegaConf.load(args.ext_cfg)
    sr = int(cfg.dataset.sample_rate)

    valid_root_dirs = cfg.dataset.valid_root_dirs if "valid_root_dirs" in cfg.dataset else cfg.dataset.root_dirs
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

    # --- select indices on rank0 then broadcast ---
    if rank == 0:
        N = len(valid_dataset)
        n = min(n_samples, N)
        indices = random.sample(range(N), n) if N > n else list(range(N))
    else:
        indices = None
    if is_dist_ready():
        obj = [indices]
        dist.broadcast_object_list(obj, src=0)
        indices = obj[0]

    # split indices per rank (round-robin)
    my_indices = indices[rank::world]

    # --- model build ---
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

        enable_inject=cfg.model.enable_inject,
        inject_mode = cfg.model.inject_mode,
        inject_f0_hz=cfg.model.inject_f0_hz,
        inject_sigma_hz=cfg.model.inject_sigma_hz,
        smooth_gate_kernel=cfg.model.smooth_gate_kernel,

        filter_activation=cfg.model.filter_activation,

        cond_dim=cfg.model.cond_dim,

        use_film=cfg.model.use_film,
        film_on_tgt=cfg.model.film_on_tgt,
        film_hidden=cfg.model.film_hidden,
        film_dropout=cfg.model.film_dropout,
        film_layernorm=cfg.model.film_layernorm,

        use_filt_head=cfg.model.use_filt_head,

        alt_permute=cfg.model.alt_permute
    ).to(device)

    if is_dist_ready():
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)


    class _DPCompat(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.module = m  # state_dict 키가 'module.xxx'로 들어와도 매칭됨

    load_target = _DPCompat(model)  # CPU든 GPU든 공통으로 안전
    # ckpt load on every rank (simplest)
    load_epoch = epoch if epoch is not None else 0
    model, _ = load_checkpoint(load_target, load_epoch, cfg.log, map_location=device, tag=tag)
    model = load_target.module

    if args.log_alpha2:
        model.reset_alpha2_stats()
        model.set_alpha2_logging(
            enable=True,
            self_attn=(not args.log_cross_only),
            cross_attn=(not args.log_self_only),
        )

    model.eval()

    # --- run ---
    local_sum, local_cnt = run_inference_on_indices(
        model=model,
        dataset=valid_dataset,
        indices=my_indices,
        device=device,
        outdir=outdir,
        sr=sr,
        cond_dim=int(cfg.model.cond_dim) if "cond_dim" in cfg.model else None
    )

    # --- aggregate timing ---
    total_sum = allreduce_sum(local_sum)
    total_cnt = allreduce_sum(float(local_cnt))
    if rank == 0 and total_cnt > 0:
        print(f"\n✅ Inference done on {int(total_cnt)} samples | avg_forward_time = {total_sum/total_cnt:.2f} ms")

    if args.log_alpha2:
        sums, cnts, tags = pack_alpha2_tensors(model, device)
        if sums is not None:
            if is_dist_ready():
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnts, op=dist.ReduceOp.SUM)
            if get_rank() == 0:
                print("\n=== α² report (mean of sum(alpha^2) per block) ===")
                for tag, s, c in zip(tags, sums.tolist(), cnts.tolist()):
                    mean_a2 = (s / max(c, 1e-9))
                    print(f"{tag:10s}  mean_sum_alpha2 = {mean_a2:.6f}  (sum={s:.4f}, cnt={c:.1f})")

    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of random valid samples")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch to load (ignored if --tag)")
    parser.add_argument("--tag", type=str, default=None, help='Checkpoint tag, e.g., "best"')
    parser.add_argument("--outdir", type=str, default="inference_valid_samples")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"],
                    help="Select device. 'auto' -> cuda if available else cpu")
    parser.add_argument("--backend", type=str, default="nccl",
                    help="DDP backend: 'nccl' for GPU, 'gloo' for CPU")
    
    parser.add_argument("--log_alpha2", action="store_true",
                        help="Log and report mean sum(alpha^2) per attention block (self & cross).")
    parser.add_argument("--log_self_only", action="store_true",
                        help="If set, log only self-attention (ignore cross).")
    parser.add_argument("--log_cross_only", action="store_true",
                        help="If set, log only cross-attention (ignore self).")
    parser.add_argument("--ext_cfg", type=str, default="config_inference.yaml")

    args = parser.parse_args()

    main(n_samples=args.n, epoch=args.epoch, tag=args.tag, outdir=args.outdir)
