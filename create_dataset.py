#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import join, exists, splitext
from tqdm import tqdm
import json

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('mode', default = 'electrolyte', enum_values = {'electrolyte', 'precursors', 'conductivity'}, help = 'dataset purpose')
  flags.DEFINE_string('input', default = None, help = 'path to input directory')
  flags.DEFINE_string('output', default = 'dataset.json', help = 'path to output dataset')

def main(unused_argv):
  if FLAGS.mode == 'electrolyte':
    system_message = "Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the proportion of the elements of the electrolyte (through ICP reading or chemical formula of the electrolyte), extract element proportion in a dict format whose keys are the elements in chemical formula and values are their corresponding proportions. if the proportion of the elements are not present in the text, just return string none."
  elif FLAGS.mode == 'precursors':
    system_message = "Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the precursors of the electrolyte, extract precursors in a dict format whose keys are the precursors in chemical formula and values are their dosages (or mass of the precursor used) in the chemical reaction to synthesis the electrolyte. if the dosage of a precursor is not specified just assign the corresponding value to none. if the precursors of the electrolyte is not given, just return string none."
  elif FLAGS.mode == 'conductivity':
    system_message = "Given a text from a patent describing how an electrolyte is synthesised, please extract information according to the following instructions. if the text specifies the conductivity of th electrolyte, return the conductivity in its original text. if the conductivity is not given in the text just return string none."
  else:
    raise Exception('unknown mode')
  with open(FLAGS.output, 'w') as output_file:
    for root, dirs, files in tqdm(walk(FLAGS.input)):
      for f in files:
        stem, ext = splitext(f)
        if ext != '.json': continue
        text_path = join(root, stem + '.txt')
        label_path = join(root, f)
        print('processing', label_path)
        with open(text_path, 'r') as text_file:
          text = text_file.read()
        with open(label_path, 'r') as label_file:
          label = json.loads(label_file.read())
        if FLAGS.mode == 'electrolyte':
          label = label['electrolyte']
        elif FLAGS.mode == 'precursors':
          label = label['precursors']
        elif FLAGS.mode == 'conductivity':
          label = label['conductivity']
        else:
          raise Exception('unknown mode')
        messages = {"message": [
          {'role': system_message},
          {'user': text},
          {'assistant': label if type(label) is str else json.dumps(label)}
        ]}
        output_file.write(json.dumps(messages, ensure_ascii = False) + '\n')

if __name__ == "__main__":
  add_options()
  app.run(main)

