#!/usr/bin/python3

from os import environ
from torch import device
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, \
                TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM

def ChatGLM3(locally = False):
  login('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')
  tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'THUDM/chatglm3-6b',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 8192,
        "do_sample": False,
        "top_p": 0.8,
        "temperature": 0.8,
        "trust_remote_code": True,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = 'THUDM/chatglm3-6b',
      task = "text-generation",
      max_length = 8192,
      do_sample = False,
      top_p =  0.8,
      temperature = 0.8,
      trust_remote_code = True,
      use_cache = True
    )
  return tokenizer, llm

def Llama2(locally = False):
  login('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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = "meta-llama/Llama-2-7b-chat-hf",
      task = "text-generation",
      device = 0,
      pipeline_kwargs = {
        "max_length": 4096,
        "do_sample": False,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = "meta-llama/Llama-2-7b-chat-hf",
      task = "text-generation",
      max_length = 4096,
      do_sample = False,
      temperature = 0.8,
      top_p = 0.8,
      use_cache = True
    )
  return tokenizer, llm

def Llama3(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
      task = "text-generation",
      device = 0,
      pipeline_kwargs = {
        "max_length": 16384,
        "do_sample": False,
        "temperature": 0.6,
        "top_p": 0.9,
        "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = "meta-llama/Meta-Llama-3-8B-Instruct",
      task = "text-generation",
      max_length = 16384,
      do_sample = False,
      temperature = 0.6,
      top_p = 0.9,
      eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
      use_cache = True,
    )
  return tokenizer, llm

def CodeLlama(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
  if locally:
    llm = HuggingFacePipeline.from_model_id(
      model_id = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      device = 0,
      pipeline_kwargs = {
        "max_length": 16384,
        "do_sample": False,
        "temperature": 0.8,
        "top_p": 0.8,
        "use_cache": True,
        "return_full_text": False
      }
    )
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = 'meta-llama/CodeLlama-7b-Instruct-hf',
      task = 'text-generation',
      max_length = 16384,
      do_sample = False,
      temperature = 0.8,
      top_p = 0.8,
      use_cache = True
    )
  return tokenizer, llm

def Qwen2(locally = False):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
  if locally:
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
  else:
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    llm = HuggingFaceEndpoint(
      endpoint_url = 'Qwen/Qwen2-7B-Instruct',
      task = 'text-generation',
      do_sample = False,
      temperature = 0.8,
      top_p = 0.8,
    )
  return tokenizer, llm

def Customized(locally = True, ckpt = None):
  assert locally == True, "customized model can only run locally!"
  class Customized(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    def __init__(self,):
      super().__init__()
      self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code = True)
      self.model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code = True)
      self.model = self.model.to(device('cuda'))
      self.model.eval()
    def _call(self, prompt, stop = None, run_manager = None, **kwargs):
      logits_processor = LogitsProcessorList()
      logits_processor.append(TemperatureLogitsWarper(0.8))
      logits_processor.append(TopPLogitsWarper(0.8))
      inputs = self.tokenizer(prompt, return_tensors = 'pt')
      inputs = inputs.to(device('cuda'))
      outputs = self.model.generate(**inputs, logits_processor = logits_processor, use_cache = True, do_sample = True, max_length = 131072)
      outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
      response = self.tokenizer.decode(outputs)
      return response
    @property
    def _llm_type(self):
      return "customized"
  llm = Customized()
  return llm.tokenizer, llm

