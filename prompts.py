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
  class Electrolyte(BaseModel):
    electrolyte: Dict[str, str] = Field(discription = "a dictionary representing an electrolyte whose keys are elements' chemical formulas and values are their proportions in float format.")
  parser = JsonOutputParser(pydantic_object = Electrolyte)
  instructions = parser.get_format_instructions()
  system_message = """Given a full text of a patent about how an electrolyte is synthesised. Extract the elements and their proportions of the electrolyte synthesised in the first example.
""" + \
instructions + \
"""
The following are several examples of how an electrolyte target is extracted from a context.

Example 1
Input context:
---------------------
The resultant powdery sulfide-based solid electrolyte was analyzed through powdery X-ray diffraction (XRD) using an X-ray diffractometer (XRD) (Smart Lab Apparatus, manufactured by Rigaku Corporation). Any other peak than the peaks derived from the raw materials was not detected. Analyzed using an ICP emission spectrometric apparatus, the composition was Li:S:P:Br:I (by mol)=1.390:1.590:0.400:0.109:0.101.
---------------------
Output electrolyte:
{"Li":1.390,"S":1.590,"P":0.400,"Br":0.109,"I":0.101}

Example 2
Input context:
---------------------
The sulfide solid electrolyte was subjected to an ICP analysis, and the molar ratio of each element was measured. The ionic conductivity and the residual ratio were measured. The results are shown in Table 1.
	TABLE 1
	
	Molar ratio of each 		
	element to phosphorus 		Ionic conductivity
	a 	b 	c 			(σ)
	(Li/P) 	(S/P) 	(Cl/P) 	a − b 	a + c 	(mS/cm)
	
Ex. 1 	5.40 	4.45 	1.70 	0.95 	7.10 	8.9
---------------------
Output electrolyte:
{"Li":5.40,"S":4.45,"Cl":1.70,"P":1.00}

Example 3:
Input context:
---------------------
(S3) The amorphized powder mixture was crystallized through thermal treatment at a temperature of about 500° C. for 4 hr, thereby yielding a solid electrolyte having an argyrodite-type crystal structure, as represented by Chemical Formula 2 below.
Li6PS5Cl  [Chemical Formula 2] 
---------------------
Output electrolyte:
{"Li":6.00,"P":1.00,"S":5.00,"Cl":1.00}
"""
  system_message = system_message.replace('{','{{')
  system_message = system_message.replace('}','}}')
  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "{patent}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['patent'])
  return template, parser

def extract_precursor_template(tokenizer):
  class Precursors(BaseModel):
    precursors: Dict[str, str] = Field(description = "a dictionary whose keys are precurs' chemical formulas and values are their dosages used in the reaction for the target electrolyte.")
  parser = JsonOutputParser(pydantic_object = Precursors)
  instructions = parser.get_format_instructions()
  system_message = """The following text is about how an electrolyte is produced. Please extract precursors of the eletrolyte and their dosages in the reaction that generates the electrolyte. Precursors are materials that participate in a chemical reaction that produces a target electrolyte.
""" \
+ instructions + \
"""
There are several examples of how a set of precursors is extracted from a context.

Example 1
Input context:
---------------------
A planetary ball mill (trade name: Classic Line P-7, manufactured by Fritsch Japan Co., Ltd.) was set up. 0.661 g of lithium sulfide, 0.914 g of diphosphorus pentasulfide, 0.164 g of bromine, and 0.261 g of iodine were weighed, put into a container (45 cc, made of zirconia) for the planetary ball mill, and further 4 g of dehydrated chlorobenzene (water content: 10 ppm or less) was put thereinto, and the container was completely sealed up. This container was set in the planetary ball mill, and driven for simultaneous mixing, stirring and grinding at a table rotation number of 500 rpm for 40 hours to prepare a sulfide-based solid electrolyte.
---------------------
Output precursors:
{"Li2S": "0.661g", "P2S5": "0.914g", "Br2": "0.164g", "I2": "0.261g"}

Example 2
Input context:
---------------------
Lithium sulfide (purity: 98.5%), phosphorus pentasulfide (manufactured by Thermophos International, purity: 99.9% or more), lithium chloride (manufactured by Sigma Aldrich Co.; purity: 99%) and an elemental sulfur (manufactured by Sigma Aldrich Co.; purity: 99.9%) were used as starting materials (hereinafter, the purity of each starting material was the same). The raw materials were mixed such that the molar ratio of lithium sulfide (Li2S), phosphorus pentasulfide (P2S5), lithium chloride (LiCl) and elemental sulfur (S) (Li2S:P2S5:LiCl:S) became 42.2:11.1:35.6:11.1. Specifically, 0.464 g of lithium sulfide, 0.591 g of phosphorus pentasulfide, 0.360 g of lithium chloride and 0.085 g of an elemental sulfur were mixed to obtain a raw material mixture.
---------------------
Output precursors:
{"Li2S": "0.464g", "P2S5": "0.591g", "LiCl": "0.360g", "S": "0.085g"}
"""
  system_message = system_message.replace('{','{{')
  system_message = system_message.replace('}','}}')
  user_message = """Extract the precursors and their dosages from following context.

context:
{context}
"""
  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context'])
  return template, parser

def extract_conductivity_template(tokenizer):
  class Conductivity(BaseModel):
    conductivity: str = Field(description = "conductivity of the electrolyte produced in the first example")
  parser = JsonOutputParser(pydantic_object = Conductivity)
  instructions = parser.get_format_instructions()
  system_message = """Given a full text of a patent about how an electrolyte is synthesised. Extract the conductivity of the electrolyte in the first example.
""" + \
instructions + \
"""
The following are several examples of how the conductivity of an electrolyte is extracted from a context.

Example 1
Input context:
---------------------
The solid electrolyte prepared in Example was subjected to compression molding to thus produce a molded measurement body (diameter of 13 mm, thickness of 0.6 mm). AC potential of 10 mV was applied to the molded body, and impedance was measured at a frequency sweep of 1×106 to 100 Hz, and thus the lithium ion conductivity of the solid electrolyte was found to be very high, specifically 2.0×10−3 S/cm. Therefore, according to the preparation method of the present invention, a solid electrolyte can have high ion conductivity.
---------------------
Output conducivity:
"2.0×10−3 S/cm"
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
  human_mesg = """For the following given context, please extract the information of an electrolyte. don't use any information given in the example.

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
