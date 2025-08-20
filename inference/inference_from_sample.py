# infer_from_rendered_longnotes.py
import os, re, time, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F  # <-- NEW
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
from collections import defaultdict

# SynthExtension imports only (충돌 없음)
from model import AudioFilterModel
from logs.log_utils import load_checkpoint
from data import SpectrogramFilterDataset  # cond=dataset 옵션일 때만 사용

def to_mono(w):
    return w.mean(0, keepdim=True) if (w.dim()==2 and w.size(0)>1) else w

@torch.no_grad()
def load_wav(path, target_sr, device):
    w, sr = torchaudio.load(path)
    w = to_mono(w)
    if sr != target_sr:
        w = torchaudio.functional.resample(w, sr, target_sr)
    return w.to(device)

def save_wav(path, x, sr):
    x = x.detach().cpu().float().numpy()
    sf.write(path, x, samplerate=int(sr))

@torch.no_grad()
def stft_mag_phase(wave, n_fft, hop, win, device):
    win_t = torch.hann_window(win, device=device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=win,
                      return_complex=True, window=win_t, center=True)
    mag = spec.abs()
    phase = spec.angle()
    return mag, phase

def load_cond_from_pickle(pickle_path, uid, device, cond_dim=None):
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)
    # saved by pandas.to_pickle in your prev script
    if hasattr(df, "to_dict"):
        df = df.to_dict(orient="list")
    uids = list(map(str, df["preset_UID"]))
    params = df["predicted_param"]
    if str(uid) not in uids:
        return None
    vec = torch.tensor(np.asarray(params[uids.index(str(uid))]),
                       dtype=torch.float32, device=device).unsqueeze(0)  # [1,D]
    if cond_dim is not None and vec.shape[-1] != cond_dim:
        d = vec.shape[-1]
        if d > cond_dim:
            vec = vec[..., :cond_dim]
        else:
            pad = torch.zeros(vec.shape[:-1] + (cond_dim - d,), device=device, dtype=vec.dtype)
            vec = torch.cat([vec, pad], dim=-1)
    return vec

def build_uid_to_cond_from_dataset(cfg, device, cond_dim=None, split="valid"):
    root_dirs = cfg.dataset.valid_root_dirs if split=="valid" else cfg.dataset.root_dirs
    ds = SpectrogramFilterDataset(
        root_dirs=root_dirs,
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
    mapping = {}
    for i in range(len(ds)):
        s = ds[i]
        uid = None
        for k in ("preset_uid","preset_UID","uid","id"):
            if k in s:
                v = s[k]
                uid = str(int(v)) if isinstance(v, (int, np.integer)) else str(v)
                break
        if uid is None or "preset_vec" not in s:
            continue
        vec = s["preset_vec"]
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        if vec.dim()==1:
            vec = vec.unsqueeze(0)
        vec = vec.to(device)
        if cond_dim is not None and vec.shape[-1] != cond_dim:
            d = vec.shape[-1]
            if d > cond_dim:
                vec = vec[..., :cond_dim]
            else:
                pad = torch.zeros(vec.shape[:-1] + (cond_dim - d,), device=device, dtype=vec.dtype)
                vec = torch.cat([vec, pad], dim=-1)
        mapping[uid] = vec
    return mapping

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_dir", type=str, required=True, help="folder containing *_long.wav and *_gt.wav")
    ap.add_argument("--ext_cfg", type=str, default='config_inference.yaml')
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--outdir", type=str, default="./out_from_long")
    ap.add_argument("--on_sec", type=float, default=1.5, help="note-on seconds in *_long.wav (mask)")
    ap.add_argument("--cond_source", type=str, default="none", choices=["none","pickle","dataset"])
    ap.add_argument("--cond_pickle", type=str, default=None, help="predicted_params.pickle path (if cond_source=pickle)")
    ap.add_argument("--split", type=str, default="valid", choices=["valid","train"], help="dataset split for cond_source=dataset")
    # ---- NEW: pad/crop target to match source length ----
    ap.add_argument("--pad_tgt_to_src", action="store_true",
                    help="Pad/crop target waveform to exactly match source length (pad on the right).")
    ap.add_argument("--log_alpha2", action="store_true",
                    help="Log and report mean sum(alpha^2) per attention block (self & cross).")
    ap.add_argument("--log_self_only", action="store_true")
    ap.add_argument("--log_cross_only", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.ext_cfg)
    sr = int(cfg.dataset.sample_rate)
    n_fft, hop, win = int(cfg.dataset.n_fft), int(cfg.dataset.hop_length), int(cfg.dataset.win_length)
    use_log_mag = bool(cfg.dataset.use_log_mag)
    cond_dim = int(cfg.model.cond_dim) if "cond_dim" in cfg.model else None

    # ----- build model -----
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

        alt_permute=cfg.model.alt_permute,
    ).to(device)

    class _DPCompat(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
    load_epoch = args.epoch if args.epoch is not None else 0
    wrapper, _ = load_checkpoint(_DPCompat(model), load_epoch, cfg.log, map_location=device, tag=args.tag)
    model = wrapper.module

    if args.log_alpha2:
        model.reset_alpha2_stats()
        model.set_alpha2_logging(
            enable=True,
            self_attn=(not args.log_cross_only),
            cross_attn=(not args.log_self_only),
        )

    model.eval()

    # ----- prepare cond lookup if needed -----
    uid2cond = {}
    if args.cond_source == "dataset":
        uid2cond = build_uid_to_cond_from_dataset(cfg, device, cond_dim=cond_dim, split=args.split)

    os.makedirs(args.outdir, exist_ok=True)

    # collect *_long.wav list
    files = [f for f in os.listdir(args.long_dir) if f.endswith("_long.wav")]
    files.sort()
    if not files:
        print("No *_long.wav found in:", args.long_dir)
        return

    total_ms, count = 0.0, 0
    meter = defaultdict(float); n=0
    for fname in files:
        uid = re.sub(r"_long\.wav$", "", fname)
        src_path = os.path.join(args.long_dir, fname)
        tgt_path = os.path.join(args.long_dir, f"{uid}_gt.wav")
        if not os.path.exists(tgt_path):
            print(f"[skip] {uid}: missing {uid}_gt.wav")
            continue

        # load wavs
        src = load_wav(src_path, sr, device)  # [1,Tsrc]
        tgt = load_wav(tgt_path, sr, device)  # [1,Ttgt]

        # ---- NEW: pad/crop target to match source length (in samples) ----
        if args.pad_tgt_to_src:
            diff = src.size(-1) - tgt.size(-1)
            if diff > 0:
                # pad right with zeros (silence)
                tgt = F.pad(tgt, (0, diff))
            elif diff < 0:
                # crop to the right to match src length
                tgt = tgt[..., :src.size(-1)]

        # features (STFT)
        src_mag, src_phase = stft_mag_phase(src, n_fft, hop, win, device)
        tgt_mag, _        = stft_mag_phase(tgt, n_fft, hop, win, device)

        # ---- NEW: if STFT frames still differ, align T by pad/crop on tgt_mag ----
        Tsrc = src_mag.shape[-1]
        Ttgt = tgt_mag.shape[-1]
        if args.pad_tgt_to_src and Ttgt != Tsrc:
            if Ttgt < Tsrc:
                pad_t = Tsrc - Ttgt
                tgt_mag = F.pad(tgt_mag, (0, pad_t))            # pad time-right
            else:
                tgt_mag = tgt_mag[..., :Tsrc]                   # crop time-right

        # note_on mask (default: on_sec seconds ON)
        Tframes = src_mag.shape[-1]
        frames_on = min(int(round(args.on_sec * sr / hop)), Tframes)
        note_on = torch.zeros(1, Tframes, device=device, dtype=src_mag.dtype)
        note_on[:, :frames_on] = 1.0

        # cond (optional)
        cond = None
        if args.cond_source == "pickle" and args.cond_pickle:
            cond = load_cond_from_pickle(args.cond_pickle, uid, device, cond_dim=cond_dim)
        elif args.cond_source == "dataset":
            cond = uid2cond.get(str(uid), None)

        # forward (single pass)
        t0 = time.perf_counter()
        pred, dbg = model(
            src_mag=src_mag,
            src_phase=src_phase,
            tgt_mag=tgt_mag,
            src_audio_length=src.shape[-1],
            note_on_mask=note_on,
            cond=cond,
            tgt_audio=tgt,
            collect_metrics=True
        )  # [1, Tsrc]
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        total_ms += ms; count += 1

        # save
        base = f"{uid}"
        save_wav(os.path.join(args.outdir, f"{base}_src.wav"),  src.squeeze(0), sr)
        save_wav(os.path.join(args.outdir, f"{base}_tgt.wav"),  tgt.squeeze(0), sr)
        save_wav(os.path.join(args.outdir, f"{base}_pred.wav"), pred.squeeze(0), sr)
        # tgt length crop for quick A/B
        m = min(pred.shape[-1], tgt.shape[-1])
        save_wav(os.path.join(args.outdir, f"{base}_pred_tgtlen.wav"), pred[..., :m].squeeze(0), sr)

        print(f"[{uid}] time={ms:.2f} ms | src_len={src.shape[-1]/sr:.2f}s | tgt_len={tgt.shape[-1]/sr:.2f}s")

        for k,v in dbg.items():
            if isinstance(v,(int,float)) and v is not None:
                meter[k]+=float(v)
        n+=1

    if count > 0:
        print(f"\n✅ Done {count} items | avg_time={total_ms/count:.2f} ms | out → {args.outdir}")

    if args.log_alpha2:
        stats = model.get_alpha2_stats()
        print("\n=== α² report (mean of sum(alpha^2) per block) ===")
        for kind in ("self", "cross"):
            for _, i, mean_a2, s, c in stats.get(kind, []):
                print(f"{kind}[{i}]  mean_sum_alpha2 = {mean_a2:.6f}  (sum={s:.4f}, cnt={c:.1f})")
    
    print("=== Averages ===")
    for k,v in meter.items():
        print(f"{k}: {v/n:.6f}")

if __name__ == "__main__":
    main()
