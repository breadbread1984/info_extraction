#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def map_rerank_prompt(tokenizer):
  messages = [
    {"role": "user", "content": """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context', 'question'])
  return template

def stuff_prompt(tokenizer):
  messages = [
    {"role": "system", "content": """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}"""},
    {"role": "user", "content": "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ["context", "question"])
  return template

def map_reduce_question_prompt(tokenizer):
  messages = [
    {"role": "system", "content": """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
______________________
{context}"""},
    {"role": "user", "content": "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context', 'question'])
  return template

def map_reduce_combine_prompt(tokenizer):
  messages = [
    {'role': "system", "content": """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
{summaries}"""},
    {'role': "user", "content": "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['summaries', 'question'])
  return template

def refine_question_template(tokenizer):
  messages = [
    {"role": "system", "content": (
    "Context information is below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the context information and not prior knowledge, "
    "answer any questions"
)},
    {"role": "user", "content": "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context_str', 'question'])
  return template

def refine_template(tokenizer):
  messages = [
    {"role": "user", "content": "{question}"},
    {"role": "system", "content": "{existing_answer}"},
    {"role": "user", "content": (
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['question', 'existing_answer', 'context_str'])
  return template
