#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from models import Llama2, Llama3, CodeLlama, Qwen2
from qa import QA
from prompts import extract_example_template, extract_electrolyte_template, extract_precursor_template, extract_conductivity_template

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('output_json', default = 'output.json', help = 'path to output json')
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama2', 'llama3', 'codellama', 'qwen2'}, help = 'model name')
  flags.DEFINE_enum('type', default = 'map_rerank', enum_values = {'stuff', 'map_reduce', 'refine', 'map_rerank'}, help = 'QA chain type')

def main(unused_argv):
  if FLAGS.model == 'llama2':
    tokenizer, llm = Llama2(FLAGS.locally)
  elif FLAGS.model == 'llama3':
    tokenizer, llm = Llama3(FLAGS.locally)
  elif FLAGS.model == 'codellama':
    tokenizer, llm = CodeLlama(FLAGS.locally)
  elif FLAGS.model == 'qwen2':
    tokenizer, llm = Qwen2(FLAGS.locally)
  else:
    raise Exception('unknown model!')
  
  example_template = extract_example_template(tokenizer)
  electrolyte_template, electrolyte_parser = extract_electrolyte_template(tokenizer)
  precursor_template, precursor_parser = extract_precursor_template(tokenizer)
  conductivity_template, conductivity_parser = extract_conductivity_template(tokenizer)
  example_chain = example_template | llm
  electrolyte_chain = electrolyte_template | llm | electrolyte_parser
  precursor_chain = precursor_template | llm | precursor_parser
  conductivity_chain = conductivity_template | llm | conductivity_parser

  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() in ['.htm', '.html']:
        loader = UnstructuredHTMLLoader(join(root, f))
      elif ext.lower() == '.txt':
        loader = TextLoader(join(root, f))
      elif ext.lower() == '.pdf':
        loader = UnstructuredPDFLoader(join(root, f), mode = 'single', strategy = "hi_res")
      else:
        raise Exception('unknown format!')
      text = ''.join([doc.page_content for doc in loader.load()])
      example = example_chain.invoke({'patent': text})
      with open('%s_example.txt' % f, 'w') as fp:
        fp.write(example)
      electrolyte = electrolyte_chain.invoke({'patent': text})
      with open('%s_electrolyte.txt' % f, 'w') as fp:
        fp.write(json.dumps(electrolyte))
      precursors = precursor_chain.invoke({'context': example})
      with open('%s_precursor.txt' % f, 'w') as fp:
        fp.write(json.dumps(precursors))
      conductivity = conductivity_chain.invoke({'patent': text})
      with open('%s_conductivity.txt' % f, 'w') as fp:
        fp.write(json.dumps(conductivity))

if __name__ == "__main__":
  add_options()
  app.run(main)

