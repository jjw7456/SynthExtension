import os
import csv
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import glob

_use_wandb = False
_writer = None
_csv_file = None
_csv_writer = None


def init_loggers(logs_cfg, csv_name: str = "loss_log.csv"):
    global _use_wandb, _writer, _csv_file, _csv_writer
    ckpt_dir = logs_cfg.ckpt_dir
    log_dir = logs_cfg.log_dir
    _use_wandb = bool(getattr(logs_cfg, "use_wandb", False))

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard
    _writer = SummaryWriter(log_dir=log_dir)

    # CSV
    csv_path = os.path.join(ckpt_dir, csv_name)
    _csv_file = open(csv_path, "w", newline="", buffering=1)
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(["step", "epoch", "loss"])

    # W&B
    if _use_wandb:
        import wandb
        wandb.init(dir=ckpt_dir)
        wandb.config.update(logs_cfg)

    print(f"[Logger initialized] ckpt_dir={ckpt_dir}, log_dir={log_dir}, use_wandb={_use_wandb}")


def log_metrics(step: int, epoch: int, loss: float):
    if _writer is None or _csv_writer is None:
        raise RuntimeError("Loggers not initialized. Call init_loggers() first.")

    _writer.add_scalar("train/loss", loss, step)
    _csv_writer.writerow([step, epoch, f"{loss:.6f}"])

    if _use_wandb:
        import wandb
        wandb.log({"train/loss": loss}, step=step)


def log_epoch(epoch: int, avg_loss: float):
    if _writer is None:
        raise RuntimeError("Loggers not initialized. Call init_loggers() first.")

    _writer.add_scalar("train/avg_loss", avg_loss, epoch)
    if _use_wandb:
        import wandb
        wandb.log({"train/avg_loss": avg_loss}, step=epoch)


def close_loggers():
    global _writer, _csv_file
    if _writer is not None:
        _writer.close()
        _writer = None
    if _csv_file is not None:
        _csv_file.close()
        _csv_file = None


def save_checkpoint(model, epoch, logs_cfg, tag=None):
    tag = f"_{tag}" if tag else ""
    filename = f"epoch{epoch}{tag}.pt"
    save_path = os.path.join(logs_cfg.ckpt_dir, filename)

    os.makedirs(logs_cfg.ckpt_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"[Saved checkpoint] : {save_path}")


def load_checkpoint(model, epoch, logs_cfg, map_location='cpu', tag=None, strip_module_prefix = False):
    ckpt_dir = logs_cfg.ckpt_dir

    tag = f"_{tag}" if tag else ""
    filename = f"epoch{epoch}{tag}.pt"

    load_path = os.path.join(ckpt_dir, filename)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path} and no fallback found.")
    
    checkpoint = torch.load(load_path, map_location=map_location)
    if strip_module_prefix:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict = False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    print(f"[Loaded checkpoint] from: {load_path}")
    return model, checkpoint.get('epoch', 0)