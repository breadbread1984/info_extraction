#!/usr/bin/python3

from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.output_parsers.regex import RegexParser

def extract_example_template(tokenizer):
  messages = [
    {"role": "system", "content": """Given a full text of a patent about how an electrolyte is produced. There is several examples of how the electrlyte is produced given in the text. please extract the original text of the first example."""},
    {"role": "user", "content": "the full text:\n\n{patent}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ["patent"])
  return template

def extract_electrolyte_template(tokenizer):
  messages = [
    {'role': 'system', 'content': 'Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the proportion of the elements of the electrolyte (through ICP reading or chemical formula of the electrolyte), extract element proportion in a dict format whose keys are the elements in chemical formula and values are their corresponding proportions. if the proportion of the elements are not present in the text, just return string none.'},
    {'role': 'user', 'content': '{context}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template

def extract_precursor_template(tokenizer):
  messages = [
    {'role': 'system', 'content': 'Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the precursors of the electrolyte, extract precursors in a dict format whose keys are the precursors in chemical formula and values are their dosages (or mass of the precursor used) in the chemical reaction to synthesis the electrolyte. if the dosage of a precursor is not specified just assign the corresponding value to none. if the precursors of the electrolyte is not given, just return string none.'},
    {'role': 'user', 'content': '{context}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template

def extract_conductivity_template(tokenizer):
  messages = [
    {'role': 'system', 'content': 'Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the conductivity of th electrolyte, return the conductivity in its original text. if the conductivity is not given in the text just return string none.'},
    {'role': 'user', 'content': '{context}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template

def extract_synthesis_template(tokenizer):
  class Synthesis(BaseModel):
    synthesis_method: str = Field(description = "a short text describing the synthesis method of the electrolyte. the context may contain milling method, milling speed, milling time, annealing temperature and annealing time.")
  parser = JsonOutputParser(pydantic_object = Synthesis)
  instructions = parser.get_format_instructions()
  system_message = """The following text is about how an electrolyte is produced. Please extract and rephrase the synthesis method into a short text which may contain milling method, milling speed, milling time, annealing temperature and annealing time.
""" + \
instructions + \
"""
The following are several examples of how the synthesis method is extracted.

Example 1
Input context:
---------------------
The inside of the pot was allowed to be an argon atmosphere. The mixture was treated (mechanical milling) for 25 hours at 370 rpm by means of a planetary ball mill, and powder (intermediate) was obtained. For the resulting intermediate, the results obtained by evaluation by XRD are shown in FIG. 1.
---------------------
Output synthesis method:
{"synthesis method": "Mechanical milling at 370 rpm/25 h"}
"""
  system_message = system_message.replace('{','{{')
  system_message = system_message.replace('}','}}')
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': '{context}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template, parser

def extract_structure_template(tokenizer):
  class Structure(BaseModel):
    crystal_system: str = Field(description = "crystal system of the electrolyte thesised in the first example. the crystal system is one among triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic")
    space_group: str = Field(description = "space group is a string describing the specific category")
  parser = JsonOutputParser(pydantic_object = Structure)
  instructions = parser.get_format_instructions()
  system_message = """Given a full text of a patent about how an eletrolyte is synthesised. Extract the structure including crystal system and space group of the electrolyte produced in the first example.
""" + \
instructions + \
"""
The following are several examples of how the structure of the electrolyte is extracted.

Example 1
Input context:
---------------------
The argyrodite crystal may be orthorhombic having Pna21 space group and having a unit cell of a=15.149, b=7.476, c=10.589 [Å]; Z=4. The argyrodite crystal also may empirically be determined for example, by X-ray diffraction spectroscopy by observing peaks around at 2θ=15.5±1°, 18±1°, 26±1°, 30.5±1°, and 32±1°.
---------------------
Output electrolyte:
{"crystal system": "orthorhombic", "space group": "Pna21"}
"""
  system_message = system_message.replace('{','{{')
  system_message = system_message.replace('}','}}')
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': "{patent}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['patent'])
  return template, parser

