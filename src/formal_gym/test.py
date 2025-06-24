from formal_gym.metaxbargrammar import GrammarParams

new_language: GrammarParams = GrammarParams(
    head_initial=True,
    spec_initial=True,
    pro_drop=False,
    proper_with_det=True,
    syllable_struct="CV",
    avg_syllables=2,
    max_consonants=3,
    verbs=3,
    nouns=3,
    propns=3,
    prons=4,
    adjs=4,
    det_def=1,
    det_indef=1,
    comps=1,
)

print("The syllable structure of our new language is", new_language.syllable_struct)
print("Here are words in our new language!")
print("Verbs:", new_language.verb_lex)
print("Nouns:", new_language.noun_lex)
print("Proper Nouns:", new_language.propn_lex)
print("Pronouns:", new_language.pron_lex)
print("Adjectives:", new_language.adj_lex)
print("Determiner (definite):", new_language.det_def_lex)
print("Determiner (indefinite):", new_language.det_indef_lex)
print("Complementizers:", new_language.comp_lex)
