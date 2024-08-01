#!/usr/bin/python3

from absl import flags, app
import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('pretrained_ckpt', default = None, help = 'path to output checkpoint of pretraining')
  flags.DEFINE_string('sft_ckpt', default = 'sft', help = 'path to output checkpoint of sft')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_float('lr', default = '2e-4', help = 'learning rate')
  flags.DEFINE_integer('epoch', default = 1, help = 'epochs')
  flags.DEFINE_integer('max_seq_length', default = 4096, help = 'max sequence length')

def main(unused_argv):
  dataset = load_dataset('json', data_files = FLAGS.dataset, split = "train")
  model = AutoModelForCausalLM.from_pretrained(FLAGS.pretrained_ckpt, trust_remote_code = True)
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_ckpt, trust_remote_code = True)
  model_training_args = TrainingArguments(
    output_dir = FLAGS.sft_ckpt,
    per_device_train_batch_size = 4,
    optim = "adamw_torch",
    logging_steps = 80,
    learning_rate = FLAGS.lr,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    num_train_epochs = FLAGS.epoch,
    save_strategy = "epoch")
  lora_peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 16, lora_alpha = 32, lora_dropout = 0.05)
  trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    max_seq_length = FLAGS.max_seq_length,
    tokenizer = tokenizer,
    args = model_training_args,
    packing = True,
    peft_config = lora_peft_config,
    dataset_text_field = "message")
  trainer.train()

if __name__ == "__main__":
  add_options()
  app.run(main)
