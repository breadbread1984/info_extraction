# Introduction

this project is to provide a tool to extract data item from patent materials

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Extract

```shell
python3 main.py --input_dir <path/to/directory/of/patents> --method (stuff|map_reduce|refine|map_rerank) [--output_json <path/to/output/json>]
```

