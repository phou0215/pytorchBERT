import torch
import spacy
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
text_1 = "A Dog Run back corner near spare bedrooms"
spacy_en = spacy.load('en_core_web_sm')

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
def nltk_tokenizer(text):
    return word_tokenize(text)

print(spacy_tokenizer(text_1))
print(nltk_tokenizer(text_1))