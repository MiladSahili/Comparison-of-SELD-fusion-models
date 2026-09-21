import json
import torch
import torch.nn as nn

from src.models.audioonly.net_seld import AudioOnlyCRNN
from src.models.visual_only_2.net_seld import VisualOnlyCRNN_2


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)

    alpha = getattr(args, 'late_fusion_alpha', 0.5)

    Net = LateFusionCRNN(
        class_num=args.class_num,
        in_channels=feature_config[args.feature]["ch"],
        alpha=alpha,
    )
    return Net


class LateFusionCRNN(nn.Module):
    """Late-Fusion SELD model.

    Reuses the exact AudioOnlyCRNN and VisualOnlyCRNN architectures and only
    fuses their Multi-ACCDOA outputs at decision level.  During training it
    returns (fused, audio_out, visual_out) so that auxiliary losses can be
    applied to each branch; during evaluation it returns only the fused output,
    making it compatible with the standard SELDValidator / validate.py.
    """

    def __init__(self, class_num, in_channels, alpha=0.5):
        super().__init__()
        self.class_num = class_num
        self.alpha = alpha

        self.audio_branch = AudioOnlyCRNN(class_num=class_num, in_channels=in_channels)
        self.visual_branch = VisualOnlyCRNN_2(class_num=class_num)

    def forward(self, x_a, x_v):
        out_a = self.audio_branch(x_a, x_v)
        out_v = self.visual_branch(x_a, x_v)

        out = self.alpha * out_a + (1.0 - self.alpha) * out_v

        if self.training:
            return out, out_a, out_v
        return out
