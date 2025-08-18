import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F
import json, re
from pathlib import Path


class SpectrogramFilterDataset(Dataset):
    def __init__(
        self,
        root_dirs,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window='hann',
        use_log_mag=True,
        use_mel=False,
        n_mel_bins=128,
        transform=None,
        use_precomputed=False,
        cache_subdir="specs",

        mode="train",                        # "train" | "valid" | "test" | "infer"
        note_mask_source="auto",             # "auto" | "file"
        note_mask_dir=None,                  # when note_mask_source=="file", folder containing masks
        auto_on_seconds=3.0,                 # for auto-mask
        auto_off_seconds=1.0,                # (정보용; 지금은 on만 쓰면 됨)
        auto_fallback_when_missing=True,      # file 못 찾으면 auto로 대체,

        # ===== NEW(parent-cache): 부모 폴더에서 프리셋 캐시 자동탐색 =====
        preset_dim=2824,
        preset_fallback_zeros=False,
        uid_regex=r"^(\d+)$",           # fid에서 UID 추출 규칙
        parent_matrix_filename="preset_cache_pred_probs/presets_matrix.pt",          # 부모 폴더 파일명
        parent_uidmap_filename="preset_cache_pred_probs/preset_uid_to_row.json",     # 부모 폴더 파일명

        # (옵션) .npy(memmap) 우선 사용하고 싶으면 파일명 지정 (없으면 무시)
        parent_npy_memmap_filename=None,    # 예: "presets_matrix.fp16.npy"
    ):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.use_log_mag = use_log_mag
        self.use_mel = use_mel
        self.n_mel_bins = n_mel_bins
        self.transform = transform
        self.use_precomputed = use_precomputed
        self.cache_subdir = cache_subdir
        self.window = self._build_window(window)

        self.mode = mode
        self.note_mask_source = note_mask_source
        self.note_mask_dir = note_mask_dir
        self.auto_on_seconds = float(auto_on_seconds)
        self.auto_off_seconds = float(auto_off_seconds)
        self.auto_fallback_when_missing = bool(auto_fallback_when_missing)

        if self.use_mel:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn=torch.hann_window,      
                power=1.0,                         
                normalized=False,
                center=True,
                pad_mode='reflect',
                n_mels=self.n_mel_bins
            )
        else:
            self.mel_transform = None

        # ===== NEW(parent-cache): 프리셋 벡터 관련 세팅 =====
        self.preset_dim = int(preset_dim)
        self.preset_fallback_zeros = bool(preset_fallback_zeros)
        self._uid_regex = re.compile(uid_regex) if uid_regex else None
        self._parent_matrix_filename = parent_matrix_filename
        self._parent_uidmap_filename = parent_uidmap_filename
        self._parent_npy_memmap_filename = parent_npy_memmap_filename

        # 부모 폴더별로 한 번만 로드해서 캐시: {parent_dir_str: (("pt"| "npy_mmap", data), uid_to_row)}
        self._per_parent_cache = {}

        # 파일 쌍 스캔 (기존 로직 유지)
        self.file_pairs = []
        for root in self.root_dirs:
            file_ids = [
                f[:-7] for f in os.listdir(root)
                if f.endswith("_gt.wav") and os.path.isfile(os.path.join(root, f))
            ]
            for fid in file_ids:
                src_path = os.path.join(root, f"{fid}.wav")
                tgt_path = os.path.join(root, f"{fid}_gt.wav")
                if os.path.isfile(src_path) and os.path.isfile(tgt_path):
                    self.file_pairs.append((src_path, tgt_path, fid, root))
                    if self.use_precomputed:
                        os.makedirs(os.path.join(root, self.cache_subdir), exist_ok=True)
        self.file_pairs.sort()

    # ===== NEW(parent-cache): fid -> uid =====
    def _fid_to_uid(self, fid):
        if self._uid_regex is None:
            return None
        m = self._uid_regex.search(str(fid))
        return int(m.group(1)) if m else None

    # ===== NEW(parent-cache): 부모 폴더 캐시 로딩 (lazy) =====
    def _parent_key(self, root: str) -> str:
        return str(Path(root).resolve().parent)

    def _ensure_parent_cache(self, root: str):
        pkey = self._parent_key(root)
        if pkey in self._per_parent_cache:
            return

        parent = Path(pkey)
        # 1) npy(memmap) 우선
        if self._parent_npy_memmap_filename:
            npy_path = parent / self._parent_npy_memmap_filename
        else:
            npy_path = None

        pt_path = parent / self._parent_matrix_filename
        map_path = parent / self._parent_uidmap_filename

        uid_to_row = None
        if map_path.is_file():
            with open(map_path, "r") as f:
                tmp = json.load(f)
                uid_to_row = {int(k): int(v) for k, v in tmp.items()}

        if uid_to_row is None:
            # 매핑이 없으면 조회 불가 → 빈 캐시
            self._per_parent_cache[pkey] = (None, None)
            return

        # memmap 사용
        if npy_path is not None and npy_path.is_file():
            mm = np.load(npy_path, mmap_mode="r")  # [N, P] float16/float32
            assert mm.shape[1] == self.preset_dim, f"preset_dim mismatch: {mm.shape[1]} vs {self.preset_dim}"
            self._per_parent_cache[pkey] = (("npy_mmap", mm), uid_to_row)
            return

        # .pt 텐서 (전체 로드)
        if pt_path.is_file():
            mat = torch.load(pt_path.as_posix(), map_location="cpu")
            assert mat.shape[1] == self.preset_dim, f"preset_dim mismatch: {mat.shape[1]} vs {self.preset_dim}"
            self._per_parent_cache[pkey] = (("pt", mat), uid_to_row)
            return

        # 아무 것도 없으면 빈 캐시
        self._per_parent_cache[pkey] = (None, None)

    # ===== NEW(parent-cache): vec 조회 =====
    def _get_preset_vec(self, root: str, fid) -> torch.Tensor:
        uid = self._fid_to_uid(fid)
        if uid is None:
            return None
        self._ensure_parent_cache(root)
        pkey = self._parent_key(root)
        pack, uid_to_row = self._per_parent_cache.get(pkey, (None, None))
        if pack is None or uid_to_row is None:
            return None
        kind, data = pack
        row = uid_to_row.get(uid)
        if row is None:
            return None
        if kind == "npy_mmap":
            # memmap → torch.float32 (한 행만 슬라이스)
            v = torch.from_numpy(np.array(data[row])).to(torch.float32)
            return v
        else:  # "pt"
            return data[row].to(torch.float32)

    def _auto_note_on_mask(self, T: int) -> torch.Tensor:
        """3s on / 1s off 규칙으로 [1, T] float 마스크 생성"""
        on_frames = int(self.auto_on_seconds * self.sample_rate / self.hop_length)
        mask = torch.zeros(T, dtype=torch.float32)
        mask[:min(on_frames, T)] = 1.0
        return mask.unsqueeze(0)  # [1, T]

    def _load_note_on_mask_from_file(self, item_path: str, T: int) -> torch.Tensor:
        """
        item_path(오디오/스펙 파일 경로) 기준으로 note_on_mask 파일을 찾는다.
        파일명 규칙은 필요에 맞춰 변경: <basename>.npy / .pt 등
        """
        if self.note_mask_dir is None:
            raise FileNotFoundError("note_mask_dir is not set but note_mask_source=='file'")

        base = os.path.splitext(os.path.basename(item_path))[0]
        cand_npy = os.path.join(self.note_mask_dir, base + ".npy")
        cand_pt  = os.path.join(self.note_mask_dir, base + ".pt")

        mask = None
        if os.path.isfile(cand_npy):
            arr = np.load(cand_npy)  # shape T 혹은 (1,T)
            mask = torch.from_numpy(arr).float()
        elif os.path.isfile(cand_pt):
            mask = torch.load(cand_pt, map_location="cpu").float()

        if mask is None:
            if self.auto_fallback_when_missing:
                return self._auto_note_on_mask(T)
            raise FileNotFoundError(f"note_on_mask not found for: {base}")

        # shape 정리: [1, T]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        # 길이 맞추기
        if mask.shape[-1] < T:
            pad = T - mask.shape[-1]
            mask = F.pad(mask, (0, pad))
        elif mask.shape[-1] > T:
            mask = mask[..., :T]
        mask.clamp_(0.0, 1.0)
        return mask  # [1, T]
    
    def _load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        return audio

    def _build_window(self, window_type):
        if window_type == 'hann':
            return torch.hann_window(self.win_length)
        elif window_type == 'hamming':
            return torch.hamming_window(self.win_length)
        else:
            return torch.ones(self.win_length)

    def _stft(self, audio):
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        mag = spec.abs().squeeze(0)
        phase = spec.angle().squeeze(0)
        mel = None
        if self.use_mel:
            mel = self.mel_transform(audio)         
            mel = torch.clamp(mel, min=1e-5)      
            if self.use_log_mag:
                mel = torch.log(mel)

        return mag, phase, mel

    def _spec_paths(self, root, fid, label):
        spec_dir = os.path.join(root, self.cache_subdir)
        mag_path = os.path.join(spec_dir, f"{fid}_{label}_mag.pt")
        phase_path = os.path.join(spec_dir, f"{fid}_{label}_phase.pt")
        mel_path = os.path.join(spec_dir, f"{fid}_{label}_mel.pt")
        return mag_path, phase_path, mel_path

    def _load_or_compute_spec(self, audio, root, fid, label):
        mag_path, phase_path, mel_path = self._spec_paths(root, fid, label)

        if self.use_precomputed and os.path.exists(mag_path) and os.path.exists(phase_path) and os.path.exists(mel_path):
            try:
                mag = torch.load(mag_path, weights_only=True)
                phase = torch.load(phase_path, weights_only=True)
                mel_mag = torch.load(mel_path, weights_only=True) if self.use_mel else None
            except:
                print(f"[Warning] Corrupt spec files for {fid}_{label}, recomputing.")
                mag, phase, mel_mag = self._stft(audio)
                if self.use_precomputed:
                    torch.save(mag, mag_path)
                    torch.save(phase, phase_path)
                    if self.use_mel and mel_mag is not None:
                        torch.save(mel_mag, mel_path)

        return mag, phase, mel_mag


    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        src_path, tgt_path, fid, root = self.file_pairs[idx]
        src_audio = self._load_audio(src_path)
        tgt_audio = self._load_audio(tgt_path)
        if self.transform:
            src_audio = self.transform(src_audio)
            tgt_audio = self.transform(tgt_audio)

        src_mag, src_phase, src_mel = self._load_or_compute_spec(src_audio, root, fid, "src")
        tgt_mag, tgt_phase, tgt_mel = self._load_or_compute_spec(tgt_audio, root, fid, "tgt")

        out = {
            "src_audio": src_audio,
            "tgt_audio": tgt_audio,
            "src_mag": src_mag,
            "src_phase": src_phase,
            "tgt_mag": tgt_mag,
            "tgt_phase": tgt_phase,
            "id": fid
        }
        if self.use_mel:
            out["src_mel"] = src_mel
            out["tgt_mel"] = tgt_mel

        # note_on_mask (기존)
        T = src_mag.shape[-1]
        if self.mode in ("train", "valid"):
            note_on_mask = self._auto_note_on_mask(T)
        else:
            if self.note_mask_source == "file":
                note_on_mask = self._load_note_on_mask_from_file(fid, T)
            else:
                note_on_mask = self._auto_note_on_mask(T)
        out["note_on_mask"] = note_on_mask

        # ===== NEW(parent-cache): preset_vec 주입 =====
        vec = self._get_preset_vec(root, fid)
        if vec is None:
            if self.preset_fallback_zeros:
                vec = torch.zeros(self.preset_dim, dtype=torch.float32)
            else:
                raise KeyError(f"[preset_vec] not found for fid='{fid}' (parent='{self._parent_key(root)}')")
        out["preset_vec"] = vec  # [preset_dim]

        return out

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/SynthExtension/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    dataset = SpectrogramFilterDataset(
        root_dirs=cfg.dataset.root_dirs,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        use_log_mag=cfg.dataset.use_log_mag,
        use_mel=True,
        n_mel_bins=cfg.dataset.n_mel_bins,
        use_precomputed=True,
        cache_subdir=cfg.dataset.cache_subdir
    )

    print(f"Precomputing spectrograms for {len(dataset)} file pairs...")

    for _ in tqdm(range(len(dataset))):
        _ = dataset[_]  # triggers __getitem__ which saves .pt if missing

    print("Spectrogram caching complete.")
