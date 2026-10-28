"""Exercise the real Trainer interface with a tiny random CPU model, not Llama."""
import json
import os
from pathlib import Path
import tempfile

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true')
import torch
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from retrain import classification_metrics, make_trainer_class


class TinyModel(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self):
        super().__init__(PretrainedConfig())
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 16)
    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=1):
        return CausalLMOutput(logits=self.head(self.embed(input_ids)[:, -1:]))


def verify():
    rows = [{'input_ids': [12, 13, 14], 'attention_mask': [1, 1, 1], 'labels': i} for i in range(6)]
    def collate(items):
        return {key: torch.tensor([item[key] for item in items]) for key in items[0]}
    with tempfile.TemporaryDirectory(prefix='llama-trainer-cpu-check-') as temp:
        args = TrainingArguments(output_dir=temp, use_cpu=True, max_steps=2,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=2, remove_unused_columns=False, label_names=['labels'],
            report_to=[], save_strategy='no', disable_tqdm=True)
        cls = make_trainer_class(Trainer, [1, 3, 5, 7, 9, 11], .05)
        trainer = cls(model=TinyModel(), args=args, train_dataset=rows, eval_dataset=rows,
                      data_collator=collate, compute_metrics=classification_metrics)
        trainer.train()
        prediction = trainer.predict(rows)
        assert prediction.predictions.shape == (6, 6)
        assert prediction.label_ids.shape == (6,)
        assert 'test_macro_f1' in prediction.metrics
    report = {'status': 'passed', 'scope': 'two-step tiny random CPU model; not Llama training',
              'trainer_compute_loss_compatible': True, 'classification_metrics_compatible': True,
              'prediction_shape': [6, 6], 'downloaded_weights_loaded': False,
              'packages_installed': False, 'network_used': False}
    (Path(__file__).parent/'trainer_interface_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify()
