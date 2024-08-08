# Introduction

this project is to provide a tool to extract knowledge graph from patent materials

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Extract Knowledge Graph

```shell
python3 main.py --model (llama3|qwen2) --input_dir <path/to/directory/of/patents> [--host <host>] [--user <user>] [--password <password>] [--database <database>] [--locally]
```

