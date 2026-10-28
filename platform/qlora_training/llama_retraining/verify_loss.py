"""Tiny CPU tensor check; this does not train or load a Llama model."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from retrain import restricted_logits


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(1, 1, 16))
    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert logits_to_keep == 1 and not use_cache
        return SimpleNamespace(logits=self.values)


def verify():
    model = ToyModel()
    tokens = [1, 3, 5, 7, 9, 11]
    inputs = torch.tensor([[12, 13]])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    logits = restricted_logits(model, inputs, torch.ones_like(inputs), tokens)
    target = torch.tensor([2])
    loss = functional.cross_entropy(logits, target, label_smoothing=.05)
    loss.backward()
    gradients = model.values.grad[0, 0]
    assert torch.isfinite(gradients).all()
    assert all(gradients[i].item() == 0 for i in range(16) if i not in tokens)
    assert gradients[tokens[2]].item() < 0
    before = loss.item()
    optimizer.step()
    after = functional.cross_entropy(restricted_logits(model, inputs, torch.ones_like(inputs), tokens),
                                     target, label_smoothing=.05).item()
    assert after < before
    report = {'status': 'passed', 'scope': 'tiny CPU tensors only, not a Llama training run',
              'selected_logits_shape': list(logits.shape), 'only_class_logits_received_gradient': True,
              'optimizer_step_reduced_toy_loss': True, 'model_weights_loaded': False,
              'packages_installed': False, 'network_used': False}
    (Path(__file__).parent/'local_loss_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify()
