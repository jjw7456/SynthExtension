import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss with spectral convergence and magnitude loss.
    """
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024,
                 mag_weight=1.0, sc_weight=1.0, use_log_mag=True):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.mag_weight = mag_weight
        self.sc_weight = sc_weight
        self.use_log_mag = use_log_mag
        self.window = torch.hann_window(win_length)

    def forward(self, x, y):
        """
        Args:
            x: [B, T] - predicted waveform
            y: [B, T] - target waveform
        Returns:
            total_loss: scalar STFT loss value
        """
        device = x.device
        window = self.window.to(device)

        x_stft = torch.stft(x, n_fft=self.fft_size, hop_length=self.hop_size, 
                            win_length=self.win_length, window=window, return_complex=True)
        y_stft = torch.stft(y, n_fft=self.fft_size, hop_length=self.hop_size, 
                            win_length=self.win_length, window=window, return_complex=True)

        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        sc_loss = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-8)

        # Magnitude Loss
        if self.use_log_mag:
            x_mag = torch.log(x_mag + 1e-7)
            y_mag = torch.log(y_mag + 1e-7)
        mag_loss = F.l1_loss(x_mag, y_mag)

        return self.sc_weight * sc_loss + self.mag_weight * mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss: sum of STFT losses at multiple resolutions.
    """
    def __init__(self, fft_sizes, hop_sizes, win_lengths,
                 mag_weight=1.0, sc_weight=1.0, use_log_mag=True):
        super().__init__()
        self.loss_modules = nn.ModuleList([
            STFTLoss(fft, hop, win, mag_weight, sc_weight, use_log_mag)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x, y):
        """
        Args:
            x: [B, T] - predicted waveform
            y: [B, T] - target waveform
        Returns:
            total STFT loss across all resolutions
        """
        total_loss = 0.0
        for loss_fn in self.loss_modules:
            total_loss += loss_fn(x, y)
        return total_loss

