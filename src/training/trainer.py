import torch
import torch.optim as optim

from src.data.dataloader import create_data_loader
from src.models.midlevel.net_seld import create_net_seld
from src.losses.adpit import MSELoss_ADPIT


class SELDTrainer(object):
    def __init__(self, args):
        self._args = args
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._data_loader = create_data_loader(self._args)
        self._net = create_net_seld(self._args)
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
