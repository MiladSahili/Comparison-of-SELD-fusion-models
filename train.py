# train.py
import yaml, argparse
from src.training.trainer import SELDTrainer
from torch.utils.tensorboard import SummaryWriter

def load_args(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(
        train_wav_txt=cfg["data"]["train_wav_txt"],
        feature_config=cfg["data"]["feature_config"],
        batch_size=cfg["data"]["batch_size"],
        train_wav_length=cfg["data"]["train_wav_length"],
        video_fps=cfg["data"]["video_fps"],
        fft_size=cfg["features"]["fft_size"],
        stft_hop_size=cfg["features"]["stft_hop_size"],
        feature=cfg["features"]["feature"],
        class_num=cfg["model"]["class_num"],
        net=cfg["model"].get("net", "crnn"),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-6),      # (1) Default = Sony-Wert
        lr_decay_step=cfg["training"].get("lr_decay_step", 10000),   # (2) NEU
        lr_decay_gamma=cfg["training"].get("lr_decay_gamma", 0.5),   # (3) NEU
        iterations=cfg["training"].get("iterations", 10000),
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        random_seed=cfg["training"].get("random_seed", 0),
        max_iter=cfg["training"]["max_iter"],
        model_save_interval=cfg["training"]["model_save_interval"],
    )

def main():
    # ── 1. Setup (VOR der Schleife) ──────────────────────
    args = load_args("/app/configs/mid_fusion.yaml")

    # Seed setzen, wie Sony
    import random, numpy as np, torch
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    trainer = SELDTrainer(args)

    writer = SummaryWriter(log_dir="/app/results/logs/mid_fusion")   # <-- HIER, vor der Schleife

    # ── 2. Trainingsschleife ─────────────────────────────
    for it in range(args.max_iter):
        trainer.receive_input()
        trainer.back_propagation()

        if it % 10 == 0:
            writer.add_scalar("Loss/train", trainer.get_loss(), it)   # schreibt rein
        if it % 50 == 0:
            print(f"iter {it} | loss {trainer.get_loss():.4f}")
        if it % args.model_save_interval == 0 and it > 0:
            trainer.save(args.checkpoint_dir, it)

    # ── 3. Aufräumen (NACH der Schleife) ─────────────────
    writer.close()                                                    # <-- erst HIER


if __name__ == "__main__":
    main()