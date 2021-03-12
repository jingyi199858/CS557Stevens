from nltk.tokenize import sent_tokenize
from nltk.book import *
from nltk.tokenize.treebank import TreebankWordDetokenizer

text11 = TreebankWordDetokenizer().detokenize(text1)
tokened_sent = sent_tokenize(text11)

print(max(tokened_sent, key=len))

