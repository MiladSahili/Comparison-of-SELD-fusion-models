# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from src.models.midlevel.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, "r") as f:
        feature_config = json.load(f)

    in_channels = feature_config[args.feature]["ch"]

    if args.net == "crnn":
        Net = AudioVisualCRNN_Early(
            class_num=args.class_num,
            in_channels=in_channels
        )
    return Net


class AudioVisualCRNN_Early(nn.Module):
    def __init__(self, class_num, in_channels, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        # 1. Visuelle Projektion auf die Frequenz-Auflösung des Audios (Standard: 64 Bins)
        vis_in_size = 2 * 6 * 37  # 444
        self.vis_to_freq = nn.Linear(vis_in_size, 64)

        # 2. Gemeinsamer Encoder (in_channels + 1, da Video als zusätzlicher Kanal angehängt wird)
        embed_size = 64
        self.shared_encoder = CNN3(
            in_channels=in_channels + 1,
            out_channels=embed_size
        )

        # 3. Rekurrente Schicht (Bekoommt jetzt 64 statt 128 Features, da bereits früh fusioniert wurde)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_xyz = nn.Linear(512, 3 * 3 * self.class_num, bias=True)

    def forward(self, x_a, x_v):
        # x_a Input-Shape: (batch, time_steps, freq_bins, mic_channels)
        x_a = x_a.transpose(2, 3)
        b, c_a, t, f = x_a.size()  # Shape: (batch, mic_channels, time_steps, freq_bins)

        # --- EARLY FUSION SCHRITT ---
        # A) Video-Vektor flachklopfen und auf Frequenzbreite (f=64) projizieren
        x_v = x_v.view(b, -1)
        x_v = self.vis_to_freq(x_v)                     # Shape: (batch, freq_bins)

        # B) Auf 4D-Tensor bringen und entlang der Zeitachse (t) duplizieren
        x_v = x_v.unsqueeze(1).unsqueeze(2)             # Shape: (batch, 1, 1, freq_bins)
        x_v = x_v.expand(-1, -1, t, -1)                 # Shape: (batch, 1, time_steps, freq_bins)

        # C) Video als zusätzlichen "Mikrofon-Kanal" direkt vor dem CNN an Audio anhängen
        x = torch.cat((x_a, x_v), dim=1)                # Shape: (batch, in_channels + 1, time_steps, freq_bins)
        # ----------------------------

        # Gemeinsame Merkmalsextraktion durch das CNN
        x = self.shared_encoder(x)
        x = torch.mean(x, dim=3)                        # Frequency-Pooling -> Shape: (batch, embed_size, time_steps_downsampled)

        # Vorbereitung für die GRU
        x = x.transpose(1, 2)                           # Shape: (batch, time_steps_downsampled, embed_size)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Output-Projektion & Interpolation auf Originalzeit
        x = self.fc_xyz(x)                              # Shape: (batch, time_steps_downsampled, 3 * 3 * class_num)
        x = interpolate(x, self.interp_ratio)
        x = x.transpose(1, 2)
        x = x.view(-1, 3, 3, self.class_num, t)

        return x


class CNN3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(4, 4))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 4))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        return x