#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from models import Llama2, Llama3, CodeLlama
from qa import QA

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('output_json', default = 'output.json', help = 'path to output json')
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama2', 'llama3', 'codellama'}, help = 'model name')
  flags.DEFINE_enum('type', default = 'stuff', enum_values = {'stuff', 'map_reduce', 'refine', 'map_rerank'}, help = 'QA chain type')

def main(unused_argv):
  content = list()
  if FLAGS.model == 'llama2':
    tokenizer, llm = Llama2(FLAGS.locally)
  elif FLAGS.model == 'llama3':
    tokenizer, llm = Llama3(FLAGS.locally)
  elif FLAGS.model == 'codellama':
    tokenizer, llm = CodeLlama(FLAGS.locally)
  else:
    raise Exception('unknown model!')
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
      
      qa = QA(FLAGS.type, tokenizer, llm, text, locally = FLAGS.locally)
      formula, _ = qa.query("what is the chemical formula of the electrolyte produced in the example?")
      materials, _ = qa.query("what are the materials used in the example?")
      conductivity, _ = qa.query("what is the conductivity of the electrolyte?")
      content.append({"patent":f, "summary": summary, "chemical formula": formula, "starting materials": materials, "conductivity": conductivity})
  with open(FLAGS.output_json, 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(content, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
  add_options()
  app.run(main)

