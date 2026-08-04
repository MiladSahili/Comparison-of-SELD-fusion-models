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
        net=cfg["experiment"].get("fusion", "crnn"),          # <-- (1) AUS experiment.fusion
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-6),
        lr_decay_step=cfg["training"].get("lr_decay_step", 10000),
        lr_decay_gamma=cfg["training"].get("lr_decay_gamma", 0.5),
        iterations=cfg["training"].get("iterations", 10000),
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        log_dir=cfg["paths"]["log_dir"],                      # <-- (2) NEU: aus Config
        random_seed=cfg["training"].get("random_seed", 0),
        max_iter=cfg["training"]["max_iter"],
        model_save_interval=cfg["training"]["model_save_interval"],
        val=cfg["validation"]["val"],
        eval=cfg["validation"]["eval"],
        val_wav_txt=cfg["validation"]["val_wav_txt"],
        eval_wav_txt=cfg["validation"]["eval_wav_txt"],
        eval_model=cfg["validation"]["eval_model"],
        threshold_config=cfg["validation"]["threshold_config"],
        eval_wav_hop_length=cfg["validation"]["eval_wav_hop_length"],
        sampling_frequency=cfg["validation"]["sampling_frequency"],
    )


def main():
    args = load_args("/app/configs/audio_only.yaml")          # <-- (3) Audio-Only-Config

    import random, numpy as np, torch
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    trainer = SELDTrainer(args)

    writer = SummaryWriter(log_dir=args.log_dir)              # <-- aus Config statt hardcoded

    for it in range(args.max_iter):
        trainer.receive_input()
        trainer.back_propagation()

        if it % 10 == 0:
            writer.add_scalar("Loss/train", trainer.get_loss(), it)
        if it % 50 == 0:
            print(f"iter {it} | loss {trainer.get_loss():.4f} | net {args.net}")
        if it % args.model_save_interval == 0 and it > 0:
            trainer.save(args.checkpoint_dir, it)

    trainer.save(args.checkpoint_dir, args.max_iter)
    writer.close()


if __name__ == "__main__":
    main()