# Copyright 2023 Sony Group Corporation.
# Modified for Transformer-based Mid-Fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from src.models.midlevel.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    if args.net == 'mid_transformer':
        Net = AudioVisualTransformerCRNN(
            class_num=args.class_num,
            in_channels=feature_config[args.feature]["ch"],
            dropout=getattr(args, 'dropout', 0.05)
        )
    else:
        raise ValueError(f"midlevel_Transformer model requested with unknown net type: {args.net}")
    return Net


class AudioVisualTransformerCRNN(nn.Module):
    def __init__(self, class_num, in_channels, dropout=0.05, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio
        self.dropout = dropout

        # ==========================================
        # 1. AUDIO ENCODER
        # ==========================================
        aud_embed_size = 64
        self.audio_encoder = CNN3(in_channels=in_channels, out_channels=aud_embed_size)

        # Audio GRU: 2-layer, bidirectional, hidden=128 -> output 256
        self.audio_gru = nn.GRU(
            input_size=aud_embed_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        d_model = 256  # 2 * 128 (bidirectional)

        # ==========================================
        # 2. SELF-ATTENTION LAYERS
        # ==========================================
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dropout=dropout,
            batch_first=True
        )
        self.self_attention = nn.TransformerEncoder(self_attn_layer, num_layers=2)

        # ==========================================
        # 3. VISUAL ENCODER
        # ==========================================
        vis_in_size = 2 * 6 * 37  # 444 (YOLO Gaussian Features)
        self.vision_encoder = nn.Sequential(
            nn.Linear(vis_in_size, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ==========================================
        # 4. TRANSFORMER CROSS-ATTENTION (MID-FUSION)
        # ==========================================
        cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(cross_attn_layer, num_layers=1)

        # ==========================================
        # 5. FFN + OUTPUT HEAD
        # ==========================================
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 3 * 3 * self.class_num)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_a, x_v):
        # --- AUDIO ZWEIG ---
        x_a = x_a.transpose(2, 3)
        b, c, t, f = x_a.size()  # input: batch_size, mic_channels, time_steps, freq_bins

        x_a = self.audio_encoder(x_a)
        x_a = torch.mean(x_a, dim=3)  # (B, 64, T_audio)  T_audio = 8

        x_a = x_a.transpose(1, 2)  # (B, T_audio, 64)
        self.audio_gru.flatten_parameters()
        (x_a, _) = self.audio_gru(x_a)  # (B, T_audio, 256)

        # Self-Attention auf der Audio-Sequenz
        x_a = self.self_attention(x_a)  # (B, T_audio, 256)

        # --- VISUAL ZWEIG ---
        x_v = x_v.view(x_v.size(0), -1)  # (B, 444)
        x_v = self.vision_encoder(x_v)  # (B, 256)
        x_v = torch.unsqueeze(x_v, dim=1)  # (B, 1, 256)

        # --- TRANSFORMER CROSS-ATTENTION (MID FUSION) ---
        x = self.transformer_decoder(tgt=x_a, memory=x_v)  # (B, T_audio, 256)

        # --- FFN + OUTPUT & RESHAPE ---
        x = self.ffn(x)  # (B, T_audio, 117)

        x = interpolate(x, self.interp_ratio)  # Interpolation auf volle 128 Frames
        x = x.transpose(1, 2)
        x = x.view(-1, 3, 3, self.class_num, t)

        return x


class CNN3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
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
