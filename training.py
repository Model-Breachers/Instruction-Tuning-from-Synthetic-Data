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
os.environ[" PJRT_DEVICE"] = "TPU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from torch.multiprocessing import set_start_method


def init_args(raw_args):
    # Training args should follow FlanT5 (Scaling Instruction-Finetuned Language Models)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="/kaggle/input/trainpaca/train.json")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(raw_args)
    return args


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name_or_path)
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


def main(raw_args=None):
    torch.set_float32_matmul_precision("high")
    args = init_args(raw_args)
    seed_everything(args.seed)
    model = LightningModel(args)

    saver = ModelCheckpoint(
        verbose=True,
        dirpath=args.output_dir,
        save_weights_only=True,
    )

    strategy = "auto"
    if args.use_fsdp:
        # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
        strategy = MyFSDPStrategy(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={T5Block},
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
            activation_checkpointing=T5Block,
            cpu_offload=True,
        )

    trainer = pl.Trainer(
        precision="32",
        accelerator="tpu",
        strategy=strategy,
        accumulate_grad_batches=1 if args.debug else args.gradient_accumulation_steps,
        default_root_dir=args.output_dir,
        gradient_clip_val=None if args.use_fsdp else 1.0,
        max_epochs=args.train_epochs,
        callbacks=[saver],
        logger=False,
        overfit_batches=10 if args.debug else 0,
    )

    trainer.fit(model)



if __name__ == "__main__":
    main()
