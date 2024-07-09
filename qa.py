#!/usr/bin/python3

from os.path import splitext, join, exists
from shutil import rmtree
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers.regex import RegexParser
from models import Llama3
from prompts import *

def get_qa_chain(chain_type, llm, tokenizer):
  assert chain_type in {"stuff", "map_reduce", "refine", "map_rerank"}
  extra_kwargs = {
    "stuff": {
      "prompt": stuff_prompt(tokenizer)
    },
    "map_reduce": {
      "question_prompt": map_reduce_question_prompt(tokenizer),
      "combine_prompt": map_reduce_combine_prompt(tokenizer)
    },
    "refine": {
      "question_prompt": refine_question_template(tokenizer),
      "refine_prompt": refine_template(tokenizer)
    },
    "map_rerank": {
      "prompt": map_rerank_prompt(tokenizer)
    }
  }
  return load_qa_chain(llm, chain_type, **extra_kwargs[chain_type])

class QA(object):
  def __init__(self, chain_type, tokenizer, llm, text, db_dir = 'db', locally = False):
    if exists(db_dir): rmtree(db_dir)
    # load pdf to vectordb
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    split_texts = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma.from_texts(
        texts = split_texts,
        embedding = embeddings,
        persist_directory = db_dir)
    self.retriever = vectordb.as_retriever()
    # create chain
    self.chain = get_qa_chain(chain_type, llm, tokenizer, **({"output_parser": RegexParser(regex=r"(.*?)\nScore: (.*)", output_keys=["answer", "score"],)} if chain_type == 'map_rerank' else {}))
  def query(self, question):
    docs = self.retriever.get_relevant_documents(question)
    res = self.chain({'input_documents': docs, 'question': question}, return_only_outputs = True)
    return res['output_text']

