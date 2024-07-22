#!/usr/bin/python3

from prompts import extract_example_template, \
                    exists_electrolyte_template, \
                    extract_electrolyte_template, \
                    extract_precursor_template, \
                    extract_conductivity_template, \
                    extract_synthesis_template, \
                    extract_structure_template

def example_chain(llm, tokenizer):
  example_template = extract_example_template(tokenizer)
  example_chain = example_template | llm
  return example_chain

def exists_chain(llm, tokenizer):
  exists_template, parser = exists_electrolyte_template(tokenizer)
  exists_chain = exists_template | llm | parser
  return exists_chain

def electrolyte_chain(llm, tokenizer):
  electrolyte_template, electrolyte_parser = extract_electrolyte_template(tokenizer)
  electrolyte_chain = electrolyte_template | llm | electrolyte_parser
  return electrolyte_chain

def precursor_chain(llm, tokenizer):
  precursor_template, precursor_parser = extract_precursor_template(tokenizer)
  precursor_chain = precursor_template | llm | precursor_parser
  return precursor_chain

def conductivity_chain(llm, tokenizer):
  conductivity_template, conductivity_parser = extract_conductivity_template(tokenizer)
  conductivity_chain = conductivity_template | llm | conductivity_parser
  return conductivity_chain

def synthesis_chain(llm, tokenizer):
  synthesis_template, synthesis_parser = extract_synthesis_template(tokenizer)
  synthesis_chain = synthesis_template | llm | synthesis_parser
  return synthesis_chain

def structure_chain(llm, tokenizer):
  structure_template, structure_parser = extract_structure_template(tokenizer)
  structure_chain = structure_template | llm | structure_parser
  return structure_chain

