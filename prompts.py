#!/usr/bin/python3

from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.output_parsers.regex import RegexParser

def extract_example_template(tokenizer):
  messages = [
    {"role": "system", "content": """Given a full text of a petent about how an electrolyte is produced. There is several examples of how the electrlyte is produced given in the text. please extract the original text of the first example."""},
    {"role": "user", "content": "the full text:\n\n{patent}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ["patent"])
  return template

def customized_template(tokenizer):
  system_mesg = """Extracting information of the electrolyte in the first example as it is written in the context.
The information of an electrolyte includes a target electrolyte, multiple corresponding precursors set each of which can independently produce the target electrolyte, the structure (crystal system and space group) of the electrolyte, ionic conductivity conductivity and synthesis method of the electrolyte.

The following are several examples of how a set of precursor is extracted from a context.

Example 1
Input context:
---------------------
A planetary ball mill (trade name: Classic Line P-7, manufactured by Fritsch Japan Co., Ltd.) was set up. 0.661 g of lithium sulfide, 0.914 g of diphosphorus pentasulfide, 0.164 g of bromine, and 0.261 g of iodine were weighed, put into a container (45 cc, made of zirconia) for the planetary ball mill, and further 4 g of dehydrated chlorobenzene (water content: 10 ppm or less) was put thereinto, and the container was completely sealed up. This container was set in the planetary ball mill, and driven for simultaneous mixing, stirring and grinding at a table rotation number of 500 rpm for 40 hours to prepare a sulfide-based solid electrolyte.
---------------------
Output precursors:
{
  "precursors": [
    {"Li2S": "0.661g", "P2S5": "0.914g", "Br2": "0.164g", "I2": "0.261g"}
  ]
}

The following are several examples of how an electrolyte target is extracted from a context.

Example 1
Input context:
---------------------
The resultant powdery sulfide-based solid electrolyte was analyzed through powdery X-ray diffraction (XRD) using an X-ray diffractometer (XRD) (Smart Lab Apparatus, manufactured by Rigaku Corporation). Any other peak than the peaks derived from the raw materials was not detected. Analyzed using an ICP emission spectrometric apparatus, the composition was Li:S:P:Br:I (by mol)=1.390:1.590:0.400:0.109:0.101.
---------------------
Output electrolyte:
{
  "electrolyte": {
    "elements": ["Li","S","P","Br","I"],
    "ratio": [1.390,1.590,0.400,0.109,0.101]
  }
}

The following are several examples of how the structure of the electrolyte is extracted.

Example 1
Input context:
---------------------
The argyrodite crystal may be orthorhombic having Pna21 space group and having a unit cell of a=15.149, b=7.476, c=10.589 [Å]; Z=4. The argyrodite crystal also may empirically be determined for example, by X-ray diffraction spectroscopy by observing peaks around at 2θ=15.5±1°, 18±1°, 26±1°, 30.5±1°, and 32±1°.
---------------------
Output electrolyte:
{
  "structure": {
    "crystal system": "orthorhombic",
    "space group": "Pna21"
  }
}

The following are several examples of how the conductivity is extracted.

Example 1
Input context:
---------------------
The solid electrolyte prepared in Example was subjected to compression molding to thus produce a molded measurement body (diameter of 13 mm, thickness of 0.6 mm). AC potential of 10 mV was applied to the molded body, and impedance was measured at a frequency sweep of 1×106 to 100 Hz, and thus the lithium ion conductivity of the solid electrolyte was found to be very high, specifically 2.0×10−3 S/cm. Therefore, according to the preparation method of the present invention, a solid electrolyte can have high ion conductivity.
---------------------
Output conducivity:
{
  "ionic conductivity": "2.0×10−3 S/cm"
}

The following are several examples of how the synthesis method is extracted.

Example 1
Input context:
---------------------
The inside of the pot was allowed to be an argon atmosphere. The mixture was treated (mechanical milling) for 25 hours at 370 rpm by means of a planetary ball mill, and powder (intermediate) was obtained. For the resulting intermediate, the results obtained by evaluation by XRD are shown in FIG. 1.
---------------------
Output synthesis method:
{
  "synthesis method": "Mechanical milling at 370 rpm/25 h"
}
"""
  system_mesg = system_mesg.replace('{','{{')
  system_mesg = system_mesg.replace('}','}}')
  human_mesg = """For the following given context, please extract the information of an electrolyte

context:

{context}

output:"""
  messages = [
    {"role": "system", "content": system_mesg},
    {"role": "user", "content": human_mesg}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ["context"])
  return template
