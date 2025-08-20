import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# --- 2D Sinusoidal Positional Encoding ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        self.d_model = d_model
        
    def forward(self, x, time_offset=0):
        """
        Args:
            x: Tensor of shape [B, D, H, W]
        Returns:
            Tensor of shape [B, D, H, W] with 2D positional encoding added
        """
        B, D, H, W = x.shape
        d_h = D // 2
        d_w = D - d_h

        # Frequency axis positional encoding
        freq_pos = torch.arange(H, device=x.device).unsqueeze(1)  # [H,1]
        div_term_h = torch.exp(
            torch.arange(0, d_h, 2, device=x.device).float() *
            (-math.log(10000.0) / d_h)
        )  # [d_h/2]
        pe_h = torch.zeros(H, d_h, device=x.device)  # [H, d_h]
        pe_h[:, 0::2] = torch.sin(freq_pos * div_term_h)
        pe_h[:, 1::2] = torch.cos(freq_pos * div_term_h)
        pe_h = pe_h.transpose(0, 1).unsqueeze(2).expand(-1, -1, W)  # [d_h, H, W]

        # Time axis positional encoding
        time_pos = torch.arange(W, device=x.device).unsqueeze(1) + time_offset  # [W,1]
        div_term_w = torch.exp(
            torch.arange(0, d_w, 2, device=x.device).float() *
            (-math.log(10000.0) / d_w)
        )  # [d_w/2]
        pe_w = torch.zeros(W, d_w, device=x.device)  # [W, d_w]
        pe_w[:, 0::2] = torch.sin(time_pos * div_term_w)
        pe_w[:, 1::2] = torch.cos(time_pos * div_term_w)
        pe_w = pe_w.transpose(0, 1).unsqueeze(1).expand(-1, H, -1)  # [d_w, H, W]

        # Combine frequency and time encodings
        pe = torch.cat([pe_h, pe_w], dim=0)  # [D, H, W]
        pe = pe.unsqueeze(0)  # [1, D, H, W]

        return x + pe


# --- CNN Encoder ---
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256, num_layers=4, norm_type = 'GN', gn_groups = 32):
        """
        CNN encoder that downsamples the input spectrogram.
        Args:
            in_channels: number of input channels (1 for magnitude)
            embed_dim: number of output channels for each conv layer
            num_layers: number of convolutional layers
        """
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_layers):
            if norm_type=='BN':
                norm = nn.BatchNorm2d(embed_dim)
            elif norm_type=='GN':
                norm = nn.GroupNorm(num_groups=min(gn_groups, embed_dim), num_channels=embed_dim)
            else:
                assert ValueError

            layers += [nn.Conv2d(c, embed_dim, kernel_size=3, stride=2, padding=1),
                       norm,
                       nn.GELU()]
                
            c = embed_dim

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):

        # DEPRECATED

        """
        Args:
            x: Tensor of shape [B, 1, F, T]
        Returns:
            x: Tensor of shape [B, L, D] where L = F'*T'
            spatial: tuple (F', T')
        """
        x = self.cnn(x)  # [B, D, F', T']
        B, D, Fp, Tp = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, Fp * Tp, D)

        print("[WARNING] THIS METHOD IS DEPRECATED.")
        return x, (Fp, Tp)


# --- Self-Attention Block (with alpha^2 logging) ---
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, log_alpha2: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # logging
        self.log_alpha2 = bool(log_alpha2)
        self.register_buffer("alpha2_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("alpha2_cnt", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, x):
        if self.log_alpha2:
            attn_out, attn_w = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
            # attn_w: [B, H, L, L]
            a2 = (attn_w * attn_w).sum(dim=-1).mean()  # scalar
            # detach to avoid autograd accumulation
            self.alpha2_sum += a2.detach().to(self.alpha2_sum.dtype)
            self.alpha2_cnt += torch.tensor(1.0, device=self.alpha2_cnt.device, dtype=self.alpha2_cnt.dtype)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# --- Cross-Attention Block (with alpha^2 logging) ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, log_alpha2: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # logging
        self.log_alpha2 = bool(log_alpha2)
        self.register_buffer("alpha2_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("alpha2_cnt", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, query, kv):
        if self.log_alpha2:
            attn_out, attn_w = self.attn(query, kv, kv, need_weights=True, average_attn_weights=False)
            # attn_w: [B, H, Lq, Lk]
            a2 = (attn_w * attn_w).sum(dim=-1).mean()  # scalar
            self.alpha2_sum += a2.detach().to(self.alpha2_sum.dtype)
            self.alpha2_cnt += torch.tensor(1.0, device=self.alpha2_cnt.device, dtype=self.alpha2_cnt.dtype)
        else:
            attn_out, _ = self.attn(query, kv, kv, need_weights=False)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# --- Audio Filter Encoder (CNN + 2D Positional Encoding + Self-Attn) ---
class AudioFilterEncoder(nn.Module):
    def __init__(self, embed_dim=256, cnn_layers=4, attn_heads=4, use_self_attn=True, n_self_attn=1, alt_permute=False):
        """
        Combines CNN encoder with 2D positional encoding and optional self-attention.
        """
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1, embed_dim=embed_dim, num_layers=cnn_layers)
        self.pos_enc2d = PositionalEncoding2D(embed_dim)
        self.use_self_attn = use_self_attn
        self.n_self_attn = n_self_attn
        if self.use_self_attn:
            self.self_attn_blocks = nn.ModuleList(
                [SelfAttentionBlock(embed_dim, n_heads=attn_heads) for _ in range(self.n_self_attn)]
            )
        self.alt_permute = alt_permute
    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] input magnitude spectrogram
        Returns:
            x: [B, L, D] encoded sequence
            spatial: (H, W) feature map size
        """
        feat = self.encoder.cnn(x)                  # [B, D, H, W]
        feat = self.pos_enc2d(feat)                 # add 2D positional encoding
        B, D, H, W = feat.shape
        if self.alt_permute:
            x = feat.permute(0, 3, 2, 1).reshape(B, H * W, D)
        else:
            x = feat.permute(0, 2, 3, 1).reshape(B, H * W, D)
        if self.use_self_attn:
            for blk in self.self_attn_blocks:
                x = blk(x)
        return x, (H, W)

class HProjector(nn.Module):
    def __init__(self, tie_across_F: bool = False, temp: float = 1.0):
        super().__init__()
        self.tie = bool(tie_across_F)
        self.temp = float(temp)
        self.register_parameter("w_logits", None)  # lazy param

    def _ensure_param(self, F, H, device, dtype):
        want_shape = (H,) if self.tie else (F, H)
        need_new = (
            self.w_logits is None
            or self.w_logits.shape != want_shape
            or self.w_logits.device != device
            or self.w_logits.dtype  != dtype
        )
        if need_new:
            w = torch.zeros(want_shape, device=device, dtype=dtype)
            self.w_logits = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, H, W]
        B, F, H, W = x.shape
        self._ensure_param(F, H, x.device, x.dtype)

        if self.tie:
            # w: [H] -> softmax over H
            w = torch.softmax(self.w_logits / self.temp, dim=-1)          # [H]
            y = torch.einsum('bfhw,h->bfw', x, w)                          # [B,F,W]
        else:
            # w: [F,H]
            w = torch.softmax(self.w_logits / self.temp, dim=-1)           # [F,H]
            y = torch.einsum('bfhw,fh->bfw', x, w)                         # [B,F,W]
        return y

class AudioFilterModel(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sample_rate: int = 22050,
        embed_dim: int = 256,
        cnn_layers: int = 4,
        attn_heads: int = 4,
        use_self_attn: bool = True,
        use_cross_attn: bool = True,
        n_self_attn:   int = 6,
        n_cross_attn:  int = 6,
        window_type: str = "hann",
        filter_activation: str = "relu",
        enable_inject: bool = True,
        inject_f0_hz: float = 130.8128,   # C3
        inject_sigma_hz: float = 40.0,
        smooth_gate_kernel: int = 3,
        # ---- NEW: injection mode ----
        inject_mode: str = "saw",         # "saw" | "noise"
        # ---- FiLM conditioning ----
        cond_dim: int = 2824,
        use_film: bool = False,
        film_on_tgt: bool = False,
        film_hidden: int = 512,
        film_dropout: float = 0.0,
        film_layernorm: bool = True,
        
        use_filt_head: bool = False,

        alt_permute: bool = False,

        use_hifigan: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win_length = win_length
        self.sr = sample_rate
        self.use_cross_attn = use_cross_attn
        self.n_self_attn = n_self_attn
        self.n_cross_attn = n_cross_attn
        self.filter_activation = filter_activation
        self.enable_inject = enable_inject
        self.inject_f0_hz = float(inject_f0_hz)
        self.inject_sigma_hz = float(inject_sigma_hz)
        self.smooth_gate_kernel = int(smooth_gate_kernel)

        self.inject_mode = inject_mode

        # --- iSTFT window as buffer (DP-safe) ---
        if window_type == "hann":
            win = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        elif window_type == "hamming":
            win = torch.hamming_window(win_length, periodic=True, dtype=torch.float32)
        else:
            win = torch.ones(win_length, dtype=torch.float32)
        self.register_buffer("window", win, persistent=True)
        self.register_buffer("eps", torch.tensor(1e-8, dtype=torch.float32), persistent=False)

        # --- Encoder / Attention / Head ---
        self.encoder = AudioFilterEncoder(embed_dim, cnn_layers, attn_heads, use_self_attn, n_self_attn, alt_permute)
        if use_cross_attn:
            self.cross_attn_blocks = nn.ModuleList(
                [CrossAttentionBlock(embed_dim, n_heads=attn_heads) for _ in range(self.n_cross_attn)]
            )

        self.use_filt_head = use_filt_head
        if use_filt_head:
            self.hproj = HProjector(tie_across_F=False, temp=1.0)  # F별로 독립 가중치
            
        self.to_filter = nn.Linear(embed_dim, n_fft // 2 + 1)

        # --- Injection (early) ---
        self.inject_alpha = nn.Parameter(torch.tensor(-3.0))  # sigmoid≈0.047
        self.pre_gate_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)

        # --- FiLM conditioning ---
        self.use_film = use_film
        self.film_on_tgt = film_on_tgt
        self.film_layernorm = film_layernorm
        self.cond_dim = cond_dim
        if use_film:
            layers = [nn.Linear(cond_dim, film_hidden), nn.GELU()]
            if film_dropout > 0:
                layers.append(nn.Dropout(film_dropout))
            layers.append(nn.Linear(film_hidden, 2 * embed_dim))  # -> [γ, β]
            self.cond_to_gb = nn.Sequential(*layers)
            if film_layernorm:
                self.cond_ln = nn.LayerNorm(cond_dim)
            else:
                self.cond_ln = None

        self.use_hifigan = use_hifigan


    def set_alpha2_logging(self, enable: bool = True, self_attn: bool = True, cross_attn: bool = True):
        # Self-attn blocks
        if self_attn and hasattr(self.encoder, "self_attn_blocks"):
            for blk in self.encoder.self_attn_blocks:
                blk.log_alpha2 = enable
        # Cross-attn blocks
        if cross_attn and getattr(self, "use_cross_attn", False) and hasattr(self, "cross_attn_blocks"):
            for blk in self.cross_attn_blocks:
                blk.log_alpha2 = enable

    def reset_alpha2_stats(self):
        if hasattr(self.encoder, "self_attn_blocks"):
            for blk in self.encoder.self_attn_blocks:
                if hasattr(blk, "alpha2_sum"):
                    blk.alpha2_sum.zero_(); blk.alpha2_cnt.zero_()
        if hasattr(self, "cross_attn_blocks"):
            for blk in self.cross_attn_blocks:
                if hasattr(blk, "alpha2_sum"):
                    blk.alpha2_sum.zero_(); blk.alpha2_cnt.zero_()

    def get_alpha2_stats(self):
        stats = {}
        # Self
        if hasattr(self.encoder, "self_attn_blocks"):
            arr = []
            for i, blk in enumerate(self.encoder.self_attn_blocks):
                if blk.alpha2_cnt.item() > 0:
                    arr.append(("self", i, (blk.alpha2_sum / blk.alpha2_cnt).item(),
                                blk.alpha2_sum.item(), blk.alpha2_cnt.item()))
            stats["self"] = arr
        # Cross
        if hasattr(self, "cross_attn_blocks"):
            arr = []
            for i, blk in enumerate(self.cross_attn_blocks):
                if blk.alpha2_cnt.item() > 0:
                    arr.append(("cross", i, (blk.alpha2_sum / blk.alpha2_cnt).item(),
                                blk.alpha2_sum.item(), blk.alpha2_cnt.item()))
            stats["cross"] = arr
        return stats
    
    # ---------- Utilities ----------
    def _harmonic_template(self, f0_hz, sigma_hz, device, dtype):
        Fbins = self.n_fft // 2 + 1
        freqs = torch.linspace(0.0, self.sr / 2.0, Fbins, device=device, dtype=dtype)
        T = torch.zeros(Fbins, device=device, dtype=dtype)
        if f0_hz <= 0.0:
            return T
        max_h = int((self.sr / 2.0) // f0_hz)
        if max_h <= 0:
            return T
        denom = 2.0 * (sigma_hz ** 2) if sigma_hz > 0 else 1e-8
        for k in range(1, max_h + 1):
            center = k * f0_hz
            roll = 1.0 / k
            T += roll * torch.exp(-((freqs - center) ** 2) / denom)
        T = T / T.max().clamp(min=1e-8)
        return T  # [F]

    # ---- NEW: noise template (non-negative so magnitude만 증가) ----
    def _noise_template(self, B, Fbins, Tframes, device, dtype):
        # Uniform [0,1], per-(B,F,T). 정규화해서 saw 템플릿과 비슷한 스케일 유지
        N = torch.rand(B, Fbins, Tframes, device=device, dtype=dtype)
        N = N / N.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        return N  # [B,F,T] in [0,1]

    def _maybe_smooth_gate(self, gate_t: torch.Tensor) -> torch.Tensor:
        k = self.smooth_gate_kernel
        if k is None or k <= 1 or (k % 2 == 0):
            return gate_t
        pad = (k - 1) // 2
        kernel = torch.ones(1, 1, k, device=gate_t.device, dtype=gate_t.dtype) / float(k)
        return F.conv1d(gate_t, kernel, padding=pad)

    def _apply_film(self, feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond is None:
            return feats
        cond = cond.to(feats.dtype)
        if self.cond_ln is not None:
            cond = self.cond_ln(cond)
        gb = self.cond_to_gb(cond)                    # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)             # [B, D], [B, D]
        feats = (1.0 + gamma.unsqueeze(1)) * feats + beta.unsqueeze(1)
        return feats
    
    def _rms(self, x: torch.Tensor, dim=None, keepdim=False):
        return (x.pow(2).mean(dim=dim, keepdim=keepdim) + float(self.eps)).sqrt()

    # ---------- Forward ----------
    def forward(
        self,
        src_mag: torch.Tensor,          # [B, F, T]
        src_phase: torch.Tensor,        # [B, F, T]
        tgt_mag: torch.Tensor = None,   # [B, F, T] (optional)
        src_audio_length: int = None,
        tgt_embed_cached: torch.Tensor = None,  # [B, L, D] (optional)
        note_on_mask: torch.Tensor = None,      # [B, T] in {0,1}
        cond: torch.Tensor = None,               # [B, P] (e.g., 2824)

        collect_metrics: bool = False,
        src_audio: torch.Tensor = None,         # [B,T] or [T] (optional, time-domain input for true RMS_in)
        tgt_audio: torch.Tensor = None,
    ) -> torch.Tensor:
        B, Fbins, Tframes = src_mag.shape
        dtype = src_mag.dtype
        device = src_mag.device

        # ====== (0) EARLY INJECTION BEFORE ENCODING ======
        if self.enable_inject:
            if note_on_mask is None:
                env = torch.ones(B, 1, Tframes, device=device, dtype=dtype)
            else:
                env = note_on_mask.view(B, 1, Tframes).to(dtype)

            energy_t = (src_mag + self.eps).mean(dim=1, keepdim=True)   # [B,1,T]
            gate_raw = self.pre_gate_conv(energy_t)                      # [B,1,T]
            gate_t = torch.sigmoid(gate_raw)                             # [B,1,T]
            gate_t = self._maybe_smooth_gate(gate_t)                     # [B,1,T]
            gate_t = gate_t * env

            alpha = torch.sigmoid(self.inject_alpha)

            if self.inject_mode == "saw":
                T_f = self._harmonic_template(
                    f0_hz=self.inject_f0_hz,
                    sigma_hz=self.inject_sigma_hz,
                    device=device,
                    dtype=dtype
                ).view(1, Fbins, 1)                                      # [1,F,1]
                inject = T_f * gate_t                                    # [B,F,T]
            elif self.inject_mode == "noise":
                N = self._noise_template(B, Fbins, Tframes, device, dtype)   # [B,F,T]
                inject = N * gate_t                                          # [B,F,T]
            elif self.inject_mode == "constant":
                # MIDI on 구간(env==1)에서 모든 주파수에 동일한 상수(=alpha)를 더함
                inject = gate_t                      # [B,F,T]

            else:
                raise ValueError(f"Unknown inject_mode: {self.inject_mode}")
            src_mag_in = src_mag + alpha * inject
        else:
            src_mag_in = src_mag

        # ====== (1) ENCODING ======
        if tgt_embed_cached is not None:
            tgt_embed = tgt_embed_cached
        else:
            tgt_input = tgt_mag if tgt_mag is not None else src_mag_in
            tgt_embed, _ = self.encoder(tgt_input.unsqueeze(1))          # [B,L,D]

        src_embed, (H,  W ) = self.encoder(src_mag_in.unsqueeze(1))             # [B,L,D]

        # ====== (1.5) FiLM (before cross-attn) ======
        if self.use_film:
            src_embed = self._apply_film(src_embed, cond)                # [B,L,D]
            if self.film_on_tgt:
                tgt_embed = self._apply_film(tgt_embed, cond)            # [B,L,D]

        # ====== (2) CROSS-ATTN ======
        if self.use_cross_attn:
            for blk in self.cross_attn_blocks:
                src_embed = blk(src_embed, tgt_embed)
            features = src_embed
        features = src_embed  # [B,L,D]

        if self.use_filt_head:
            features = features.view(B, H, W, -1)            # [B,H,W,D]
            filt = self.to_filter(features)                  # [B,H,W,F]  (여기서 D->F 매핑은 네가 이미 구현)
            filt = filt.permute(0, 3, 1, 2)                  # [B,F,H,W]
            filt = self.hproj(filt)                          # [B,F,W]  ← do_something
            filt = F.interpolate(filt, size=src_mag.shape[-1],
                                mode="linear", align_corners=False)  # [B,F,T] 
        else:
            filt = self.to_filter(features)                  # [B,L,F]
            filt = filt.permute(0, 2, 1)                     # [B,F,L]
            #filt = F.adaptive_avg_pool1d(filt, src_mag.shape[-1])  # [B, F, T]
            filt = F.interpolate(filt, size=src_mag.shape[-1], mode="linear", align_corners=False)  # [B,F,T]

        if self.filter_activation == "relu":
            filt = F.relu(filt)
        elif self.filter_activation == "sigmoid":
            filt = torch.sigmoid(filt)
        elif self.filter_activation == "tanh":
            filt = torch.tanh(filt)
    
        mag_filtered = src_mag_in * filt



        dbg = None
        if collect_metrics:
            with torch.no_grad():
                # on-마스크
                if note_on_mask is not None:
                    env = note_on_mask.view(src_mag.size(0), 1, src_mag.size(-1)).to(filt.dtype)  # [B,1,T]
                else:
                    env = torch.ones(src_mag.size(0), 1, src_mag.size(-1), device=filt.device, dtype=filt.dtype)

                # 마스크 평균/에너지(on 구간)
                filt_mean_on = ((filt * env).sum(dim=(1,2)) / (env.sum(dim=(1,2)) + self.eps)).mean()  # scalar
                filt_ms_on   = (((filt ** 2) * env).sum(dim=(1,2)) / (env.sum(dim=(1,2)) + self.eps)).mean()

                # 게이트 평균(on 구간) – enable_inject일 때만 의미 있음
                gate_mean = None
                try:
                    gate_mean = ((gate_t * env).sum(dim=(1,2)) / (torch.full((env.shape[0],), float(env.shape[2]), device=env.device, dtype=env.dtype) + self.eps)).mean().item()
                except NameError:
                    pass

                inject_alpha = torch.sigmoid(self.inject_alpha).item()

                dbg = {
                    "filt_mean_on": float(filt_mean_on.item()),
                    "filt_mean_sq_on": float(filt_ms_on.item()),
                    "gate_mean": gate_mean,
                    "inject_alpha": inject_alpha,
                }

        if self.use_hifigan:
            return mag_filtered
        
        # ====== (5) PHASE + iSTFT ======
        real = mag_filtered * torch.cos(src_phase)
        imag = mag_filtered * torch.sin(src_phase)
        spec_filtered = torch.complex(real, imag)

        audio_out = torch.istft(
            spec_filtered,
            n_fft=self.n_fft,
            hop_length=self.hop,
            length=src_audio_length,
            window=self.window.to(real.dtype)
        )

        if collect_metrics:
            with torch.no_grad():
                # on 구간 샘플 수(대략): frames_on * hop
                if note_on_mask is not None:
                    frames_on = note_on_mask.sum(dim=-1).to(torch.float)  # [B]
                    samps_on = (frames_on * self.hop).clamp(max=audio_out.shape[-1])
                    samps_on = int(samps_on.mean().item())  # 배치 평균값 사용
                else:
                    samps_on = audio_out.shape[-1]

                # 출력 RMS(on)
                rms_out_on = self._rms(audio_out[..., :samps_on]).item()

                # 입력 RMS(on): tgt_audio가 넘어오면 실제 time-domain 사용
                rms_in_on = None
                if tgt_audio is not None:
                    x_time = tgt_audio
                    if isinstance(x_time, torch.Tensor) and x_time.dim() == 2:
                        x_time = x_time.mean(dim=0)  # [T]로 단순화(모노)
                    rms_in_on = self._rms(x_time[..., :samps_on]).item()

                dbg.update({
                    "samps_on": int(samps_on),
                    "rms_out_on": rms_out_on,
                    "rms_in_on": rms_in_on,
                    "rms_ratio": (rms_out_on / (rms_in_on + 1e-8)) if rms_in_on is not None else None,
                })
                
        if collect_metrics:
            return audio_out, dbg
        return audio_out
