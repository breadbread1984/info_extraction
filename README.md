# Introduction

this project is to provide a tool to extract data item from patent materials

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Extract

```shell
python3 main.py --model (llama3|qwen2|codellama|codeqwen|customized) --mode (electrolyte|precursors|conductivity) --input_dir <path/to/directory/of/patents> [--output_dir <path/to/output/directory>] [--ckpt <path/to/customized/ckpt>]
```

# finetune

to improve the LLM on ability to extract electrolyte related information, we use supervised finetuning to moderate pretrain LLM's behavior.

## generate dataset

Generate dataset for extracting electrolyte

```shell
python3 create_dataset.py --mode electrolyte --input <path/to/labeled/patnet> --output <path/to/output>
```

Generate dataset for extracting precursors

```shell
python3 create_dataset.py --mode precursors --input <path/to/labeled/patent> --output <path/to/output>
```

Generate dataset for extracting conductivity

```shell
python3 create_dataset.py --mode conductivity --input <path/to/labeled/patent> --output <path/to/output>
```

## finetuning LLM

```shell
python3 finetune.py --pretrained_ckpt <hugging/face/model/id> --sft_ckpt <path/to/ckpt> --dataset <path/to/dataset> --device (cuda|cpu)
```
