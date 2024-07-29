#!/usr/bin/python3

from absl import flags, app
import pandas as pd
from shutil import rmtree
from os import mkdir
from os.path import exists, join
from huggingface_hub import login
from transformers import AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to input csv')
  flags.DEFINE_string('output_dir', default = 'examples', help = 'path to output directory')

def Qwen2():
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
  llm = HuggingFacePipeline.from_model_id(
    model_id = 'Qwen/Qwen2-7B-Instruct',
    task = 'text-generation',
    device = 0,
    pipeline_kwargs = {
      "max_length": 131072,
      "do_sample": False,
      "temperature": 0.8,
      "top_p": 0.8,
      "use_cache": True,
      "return_full_text": False
    }
  )
  return tokenizer, llm

def extract_example_template(tokenizer):
  messages = [
    {"role": "system", "content": """Given a full text of a patent about how an electrolyte is produced. There is several examples of how the electrlyte is produced given in the text. please extract the original text of the first example."""},
    {"role": "user", "content": "the full text:\n\n{patent}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ["patent"])
  return template

def example_chain(llm, tokenizer):
  example_template = extract_example_template(tokenizer)
  example_chain = example_template | llm
  return example_chain

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  tokenizer, llm = Qwen2()
  example_chain_ = example_chain(llm, tokenizer)
  patents = pd.read_csv(FLAGS.input_csv)
  for i in patents.index:
    patent = sheet.iloc[i]['Description']
    example = example_chain_.invoke({'patent': patent})
    with open(join(FLAGS.output_dir, str(i) + '.txt'), 'w') as f:
      f.write(example)

if __name__ == "__main__":
  add_options()
  app.run(main)

