#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from prompts import extract_triplets_template, cypher_generation_template, entity_generation_template, triplets_qa_template
from models import Llama3, Qwen2, Llama3_FA2, Qwen2_FA2, GPT4O

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('host', default = 'bolt://localhost:7687', help = 'host')
  flags.DEFINE_string('user', default = 'neo4j', help = 'user name')
  flags.DEFINE_string('password', default = 'neo4j', help = 'password')
  flags.DEFINE_string('database', default = 'neo4j', help = 'database')
<<<<<<< HEAD
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama3', 'llama3_fa2', 'qwen2', 'qwen2_fa2','gpt4o'}, help = 'model name')
=======
  flags.DEFINE_enum('model', default = 'llama3_fa2', enum_values = {'llama3', 'llama3_fa2', 'qwen2', 'qwen2_fa2'}, help = 'model name')
>>>>>>> 2213a8753a535ddf97023c942ec06fe37aa49c3b

def main(unused_argv):
  if FLAGS.model == 'llama3':
    tokenizer, llm = Llama3(FLAGS.locally)
  elif FLAGS.model == 'qwen2':
    tokenizer, llm = Qwen2(FLAGS.locally)
  elif FLAGS.model == 'llama3_fa2':
    tokenizer, llm = Llama3_FA2(True)
  elif FLAGS.model == 'qwen2_fa2':
    tokenizer, llm = Qwen2_FA2(True)
  elif FLAGS.model == 'gpt4o':
    llm = GPT4O(False)
  else:
    raise Exception('unknown model!')

  if FLAGS.model != 'gpt4o':
    prompt, _ = extract_triplets_template(tokenizer)
    graph_transformer = LLMGraphTransformer(
      llm = llm, prompt = prompt,
      #allowed_nodes = ['electrolyte', 'conductivity', 'precursor', 'crystal_system_of_electrolyte', 'space_group_of_electrolyte'],
      #allowed_relationships = ['has_conductivity', 'has_precursor', 'has_crystal_system', 'has_space_group'],
      node_properties = False, relationship_properties = False
    )
  else:
    graph_transformer = LLMGraphTransformer(
      llm = llm,
      node_properties = True, relationship_properties = True
    )
  neo4j = Neo4jGraph(url = FLAGS.host, username = FLAGS.user, password = FLAGS.password, database = FLAGS.database)

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
      docs = loader.load()
      graph = graph_transformer.convert_to_graph_documents(docs)
      neo4j.add_graph_documents(graph)

if __name__ == "__main__":
  add_options()
  app.run(main)

