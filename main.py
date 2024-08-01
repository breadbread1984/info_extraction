#!/usr/bin/python3

from shutil import rmtree
from os import walk, mkdir
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from models import Llama2, Llama3, CodeLlama, Qwen2, Customized
from chains import example_chain, electrolyte_chain, precursor_chain, conductivity_chain, synthesis_chain, structure_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to output directory')
  flags.DEFINE_enum('model', default = 'customized', enum_values = {'llama2', 'llama3', 'codellama', 'qwen2', 'customized'}, help = 'model name')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint')
  flags.DEFINE_enum('mode', default = 'electrolyte', enum_values = {'electrolyte', 'precursors', 'conductivity'}, help = 'mode')

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
  elif FLAGS.model == 'customized':
    tokenizer, llm = Customized(FLAGS.locally, FLAGS.ckpt)
  else:
    raise Exception('unknown model!')
  
  example_chain_ = example_chain(llm, tokenizer)
  if FLAGS.mode == 'electrolyte':
    chain = electrolyte_chain(llm, tokenizer)
  elif FLAGS.mode == 'precursors':
    chain = precursor_chain(llm, tokenizer)
  elif FLAGS.mode == 'conductivity':
    chain = conductivity_chain(llm, tokenizer)
  else:
    raise Exception('unknow mode!')

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
      example = example_chain_.invoke({'patent': text})
      output = chain.invoke({'context': example})
      with open(join(FLAGS.output_dir, '%s_meta.txt' % splitext(f)[0]), 'w') as fp:
        fp.write(output)

if __name__ == "__main__":
  add_options()
  app.run(main)

