import random
from metaxbargrammar import GrammarParams, SyncGrammarParams, XBarGrammar
from prompt import scfg_prompt

left_params = GrammarParams(
    head_initial=True,
    spec_initial=True,
    pro_drop=False,
    proper_with_det=False,
    syllable_struct=None,
    max_consonants=3,
    avg_syllables=2,
    verbs=10,
    nouns=10,
    propns=5,
    prons=3,
    adjs=10,
    det_def=3,
    det_indef=3,
    comps=4,
    seed=3,
)

right_params = GrammarParams(
    head_initial=True,
    spec_initial=True,
    pro_drop=False,
    proper_with_det=False,
    syllable_struct=None,
    max_consonants=1,
    avg_syllables=2,
    verbs=10,
    nouns=10,
    propns=5,
    prons=3,
    adjs=10,
    det_def=3,
    det_indef=3,
    comps=4,
    seed=3,
)

# Create SCFG Object, turn it into string
sync_params = SyncGrammarParams(left=left_params, right=right_params)
grammar_str = sync_params.as_cfg_str()

# Prompts to test
prompt_og = f'''You will be presented with a synchronous context-free grammar for two languages. 
    Each grammar starts with a special `S` symbol, which yields rules of the form `A -> <B C, D E>`, 
    where `A` is a non-terminal symbol, `B` and `C` are non-terminal symbols in the first language, and 
    `D` and `E` are non-terminal symbols in the second language; or `A -> <b, d>`, where `b` is a terminal 
    symbol in  the first language and `d` is a terminal symbol in the second language. The grammar may also
    contain phonetically-null symbols, which always start with `∅`; these lexical items are presented
    in the parse trees of sentences but do not appear in the final strings. You will be presented with
    a sentence from one of the languages defined by the grammar. Your job is to use the grammar to
    translate the sentence from that language into the other. You will be graded the exact string
    accuracy of the provided translation. Do not include any phonetically-null symbols in your 
    final answer, though it may be helpful to reason about where they would appear when working through
    the translation. You can reason about the translation as an intermediate step, but you must end your 
    response with the phrase `Final Answer: <translation>`.'''

simple_prompt = f'''You are a translation agent. You will take two sets of production rules corresponding to a
synchronous context-free grammar (two context free grammars). You will then take the given string from one of the grammars
and follow the rules in order to translate it to another context free grammar. You must end your 
response with the phrase `Final Answer: <translation>`'''

prompt_with_example = f'''A synchronous context-free grammar is like a regular context-free grammar (CFG), but 
instead of generating one sentence, it generates pairs of related sentences — usually in two different languages 
or structures. It defines how to build two strings in parallel using the same rules, making sure that their structure 
stays aligned. Suppose you're translating between English and French. An SCFG rule might look like this:
<example>
S → (NP VP, NP VP)
NP → (the dog, le chien)
VP → (barks, aboie)
</example>
This SCFG can generate the pair:
"the dog barks" / "le chien aboie"

You are an expert translator that can provide highly accurate translations from one language to another with the help of a
synchronous context-free grammar (SCFG). You are given the SCFG and a sentence from one language. You follow the rules in the
grammar in order to precisely translate the given sentence into the other language. IMPORTANT: You must end your 
response with the phrase `Final Answer: <translation>`
'''

# Generate the sample sentence
grammar = XBarGrammar.from_params(left_params)
sample = grammar.generate_tree()
sentence = sample["string"]  # This is the generated sentence
print("\n")

print(scfg_prompt(grammar_str=grammar_str, lhs=sentence, prompt=prompt_og), "\n\n")
print(scfg_prompt(grammar_str=grammar_str, lhs=sentence, prompt=simple_prompt), "\n\n")
print(scfg_prompt(grammar_str=grammar_str, lhs=sentence, prompt=prompt_with_example), "\n\n")