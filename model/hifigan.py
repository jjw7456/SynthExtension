# mel_vocoder.py
import torch
import torch.nn as nn
import torchaudio.transforms as T
from huggingface_hub import hf_hub_download
from speechbrain.lobes.models.HifiGAN import HifiganGenerator

class MelProjector(nn.Module):
    """
    Linear magnitude [B,F,T] -> 80-mel [B,80,T]
    파라미터는 speechbrain/tts-hifigan-libritts-22050Hz 설정에 맞춤.
    """
    def __init__(self,
                 sample_rate=22050, n_fft=1024, n_mels=80,
                 fmin=0.0, fmax=8000.0,
                 norm="slaney", mel_scale="slaney",
                 compression=True,     # log-compression
                 minmax_norm=True):    # per-utt min-max
        super().__init__()
        self.mel_scale = T.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=fmin, f_max=fmax,
            n_stft=n_fft // 2 + 1,
            norm=norm, mel_scale=mel_scale,
        )
        self.compression = bool(compression)
        self.minmax_norm = bool(minmax_norm)

    def forward(self, mag_lin: torch.Tensor) -> torch.Tensor:
        mel = self.mel_scale(mag_lin)  # [B,80,T]
        if self.compression:
            mel = torch.log(mel.clamp_min(1e-5))
        if self.minmax_norm:
            mel_min = mel.amin(dim=(1,2), keepdim=True)
            mel_max = mel.amax(dim=(1,2), keepdim=True)
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        return mel


class HiFiGANVocoder(nn.Module):
    """
    HF에서 프리트레인 로드. trainable 토글 지원.
    입력: mel [B,80,T] -> 출력: audio [B,1,time]
    """
    def __init__(self,
                 repo_id="speechbrain/tts-hifigan-libritts-22050Hz",
                 ckpt_name="generator.ckpt",
                 trainable=True):
        super().__init__()
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
        self.generator = HifiganGenerator(
            in_channels=80, out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
            resblock_kernel_sizes=[3,7,11],
            upsample_kernel_sizes=[16,16,4,4],
            upsample_initial_channel=512,
            upsample_factors=[8,8,2,2],
            inference_padding=5,
            cond_channels=0,
            conv_post_bias=True,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        self.generator.load_state_dict(state, strict=True)
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        self.generator.train(trainable)
        for p in self.generator.parameters():
            p.requires_grad = trainable

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.generator(mel)  # [B,1,samples]
