import shutil
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

import argparse
import functools
import os

import pytorch_lightning as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import (
    MixedPrecision,
    FullyShardedDataParallel,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
from transformers.models.t5.modeling_t5 import T5Block

from data_loading import TextToTextDataset
class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        print(dict(orig_state_dict=len(self.model.state_dict())))
        if self.hparams.use_lora:
            # https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, peft_config)
        if self.hparams.use_compile:
            self.model = torch.compile(self.model)
        if self.hparams.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = self.trainer.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # noinspection PyTypeChecker
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            relative_step=False,
        )
        return [optimizer]

    def train_dataloader(self):
        dataset = TextToTextDataset(
            path=self.hparams.data_path,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            tokenizer=self.tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
        )

def test_model(
    path: str ,
    prompt: str = "",
    max_length: int = 160,
    device: str = "cuda",
):
    if not prompt:
        prompt = "What does DNA stand for?"

    model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    seed_everything(model.hparams.seed)
    with torch.inference_mode():
        model.model.eval()
        model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model.model.generate(
            input_ids=input_ids, max_length=max_length, do_sample=True
        )

    print(tokenizer.decode(outputs[0]))

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def export_checkpoint(path: str, path_out: str):
    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(path_out)
    model.tokenizer.save_pretrained(path_out)


def export_to_hub(path: str , repo: str , temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)
    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


"""
huggingface-cli login

p inference.py export_to_hub \
--path "outputs_unclean/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-xl

p inference.py export_to_hub \
--path "outputs/model/xxl/epoch=0-step=203.ckpt" \
--repo declare-lab/flan-alpaca-xxl

p inference.py export_to_hub \
--path "outputs/model_gpt4all/xl/epoch=0-step=6838.ckpt" \
--repo declare-lab/flan-gpt4all-xl

p inference.py export_to_hub \
--path "outputs/model_sharegpt/xl/epoch=0-step=4485.ckpt" \
--repo declare-lab/flan-sharegpt-xl

p inference.py export_to_hub \
--path "outputs/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-gpt4-xl

"""


if __name__ == "__main__":
    Fire()
