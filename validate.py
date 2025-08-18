import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
import re

from data import SpectrogramFilterDataset
from model import AudioFilterModel
from model.loss import MultiResolutionSTFTLoss
from logs.log_utils import load_checkpoint, save_checkpoint


def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            src_audio = batch["src_audio"].to(device)
            tgt_audio = batch["tgt_audio"].to(device).squeeze(1)
            src_mag   = batch["src_mag"].to(device)
            src_phase = batch["src_phase"].to(device)
            tgt_mag   = batch["tgt_mag"].to(device)

            pred_audio = model(
                src_mag,
                src_phase,
                tgt_mag,
                src_audio_length=src_audio.shape[-1]
            )

            min_len = min(pred_audio.shape[-1], tgt_audio.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            tgt_audio  = tgt_audio[..., :min_len]

            loss = criterion(pred_audio, tgt_audio)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def extract_epoch_from_filename(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else None


def main():
    # === Load configuration ===
    cfg = OmegaConf.load("default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    # === TensorBoard writer for validation
    val_log_dir = os.path.join(cfg.log.log_dir, "valid")
    os.makedirs(val_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=val_log_dir)
    '''
    # === Dataset ===
    valid_dataset = SpectrogramFilterDataset(
        root_dirs=cfg.dataset.valid_root_dirs,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
        use_log_mag=cfg.dataset.use_log_mag,
        use_precomputed=cfg.dataset.use_precomputed,
        cache_subdir=cfg.dataset.cache_subdir
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    # === Loss ===
    criterion = MultiResolutionSTFTLoss(
        fft_sizes=cfg.loss.fft_sizes,
        hop_sizes=cfg.loss.hop_sizes,
        win_lengths=cfg.loss.win_lengths,
        mag_weight=cfg.loss.mag_weight,
        sc_weight=cfg.loss.sc_weight,
        use_log_mag=cfg.loss.use_log_mag
    ).to(device)

    # === Find all checkpoints with epoch info
    checkpoint_paths = glob.glob(os.path.join(cfg.log.ckpt_dir, "*.pt"))
    checkpoint_info = []

    '''
    for path in checkpoint_paths:
        epoch = extract_epoch_from_filename(os.path.basename(path))
        if epoch is not None:
            checkpoint_info.append((epoch, path))

    checkpoint_info.sort(key=lambda x: x[0])  # sort by epoch
    print(f"Found {len(checkpoint_info)} checkpoints with valid epoch.")

    best_loss = float("inf")
    best_epoch = -1
    best_model = None
    '''

    checkpoint_info = [(230, f'/dataset/synth_extension/logs/checkpoints/epoch230.pt')]
    for epoch, ckpt_path in checkpoint_info:
        # === Optional tag, e.g., "step300"
        tag = None

        print(f"\n[Epoch {epoch}] Evaluating: {os.path.basename(ckpt_path)}")

        # === Load model and checkpoint ===
        model = AudioFilterModel(
            n_fft=cfg.dataset.n_fft,
            hop_length=cfg.dataset.hop_length,
            win_length=cfg.dataset.win_length,
            window_type=cfg.dataset.window,
            embed_dim=cfg.model.embed_dim,
            cnn_layers=cfg.model.cnn_layers,
            attn_heads=cfg.model.attn_heads,
            use_self_attn=cfg.model.use_self_attn,
            use_cross_attn=cfg.model.use_cross_attn,
            filter_mode=cfg.filter.mode,
            filter_activation=cfg.filter.activation
        ).to(device)
        model = nn.DataParallel(model, device_ids=[0,1], output_device=0)

        model, _ = load_checkpoint(model, epoch, cfg.log, map_location=device, tag=tag)

        val_loss = validate(model, criterion, valid_loader, device)
        #writer.add_scalar("valid/epoch_loss", val_loss, epoch)

        print(f"[Epoch {epoch}] Validation loss: {val_loss:.6f}")
        '''
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = model

    # === Save best model using save_checkpoint
    if best_model is not None:
        save_checkpoint(best_model, best_epoch, cfg.log, tag="best")

    print("\n✅ Validation complete.")
    print(f"🏆 Best model: epoch{best_epoch} (loss: {best_loss:.6f}) → saved as 'epoch{best_epoch}_best.pt'")
    
    writer.close()
    '''

if __name__ == "__main__":
    main()
