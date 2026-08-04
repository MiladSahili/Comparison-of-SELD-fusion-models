import torch
import torch.optim as optim

from src.data.dataloader import create_data_loader
from src.losses.adpit import MSELoss_ADPIT


def _create_net(args):
    """Waehlt die Modell-Implementierung anhand von args.net (aus experiment.fusion)."""
    if args.net in ('crnn', 'mid'):           # Mid Fusion (bisher)
        from src.models.midlevel.net_seld import create_net_seld
    elif args.net == 'audio_only':            # Audio-Only
        from src.models.audioonly.net_seld import create_net_seld
    elif args.net in ('late_crnn', 'late'):   # Late Fusion
        from src.models.latelevel.net_seld import create_net_seld
    elif args.net == 'early':                 # Early Fusion (später)
        from src.models.earlylevel.net_seld import create_net_seld
    else:
        raise ValueError(f"Unbekannter net-Typ: {args.net}")
    return create_net_seld(args)


class SELDTrainer(object):
    def __init__(self, args):
        self._args = args
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._data_loader = create_data_loader(self._args)
        self._net = _create_net(self._args)
        self._net.to(self._device)
        self._net.train()
        self._criterion = MSELoss_ADPIT()
        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._args.lr,
            weight_decay=self._args.weight_decay
        )

    def receive_input(self):
        _input_a, _input_v, _label, _ = next(iter(self._data_loader))
        self._input_a = _input_a.to(self._device)
        self._input_v = _input_v.to(self._device)
        self._label = _label.to(self._device)

    def back_propagation(self):
        self._net.train()
        self._optimizer.zero_grad()
        self._output = self._net(self._input_a, self._input_v)

        if isinstance(self._output, tuple):
            # Late Fusion: (kombiniert, audio_branch, visual_branch)
            out, out_a, out_v = self._output
            self._loss = self._criterion(out, self._label) \
                + 0.5 * (self._criterion(out_a, self._label)
                         + self._criterion(out_v, self._label))
        else:
            # Mid Fusion / Audio-Only: unveraendert
            self._loss = self._criterion(self._output, self._label)

        self._loss.backward()
        self._optimizer.step()

    def save(self, checkpoint_dir, iteration):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = f"{checkpoint_dir}/params_{iteration:07}.pth"
        torch.save({
            'model_state_dict': self._net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, path)
        print(f"save checkpoint to {path}")

    def get_loss(self):
        return self._loss.cpu().detach().numpy()