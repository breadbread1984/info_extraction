#!/usr/bin/python3

from shutil import rmtree
from os import walk, mkdir
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from models import Llama2, Llama3, CodeLlama, Qwen2
from chains import example_chain, exists_chain, electrolyte_chain, precursor_chain, conductivity_chain, synthesis_chain, structure_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to output directory')
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama2', 'llama3', 'codellama', 'qwen2'}, help = 'model name')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
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
  
  example_chain_ = example_chain(llm, tokenizer)
  exists_chain_ = exists_chain(llm, tokenizer)
  electrolyte_chain_ = electrolyte_chain(llm, tokenizer)
  precursor_chain_ = precursor_chain(llm, tokenizer)
  conductivity_chain_ = conductivity_chain(llm, tokenizer)
  synthesis_chain_ = synthesis_chain(llm, tokenizer)
  structure_chain_ = structure_chain(llm, tokenizer)

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
      output = dict()
      example = example_chain_.invoke({'patent': text})
      electrolyte = electrolyte_chain_.invoke({'patent': text})
      output.update(electrolyte)
      exist_example = exists_chain_.invoke({'context': example})
      output.update(exist_example)
      precursors = precursor_chain_.invoke({'context': example})
      output.update(precursors)
      conductivity = conductivity_chain_.invoke({'patent': text})
      output.update(conductivity)
      synthesis = synthesis_chain_.invoke({'context': example})
      output.update(synthesis)
      structure = structure_chain_.invoke({'patent': text})
      output.update(structure)
      output['example'] = example
      with open(join(FLAGS.output_dir, '%s_meta.txt' % splitext(f)[0]), 'w') as fp:
        fp.write(json.dumps(output, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
  add_options()
  app.run(main)

