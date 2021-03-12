from collections import Counter
from nltk.tokenize import *
from nltk.stem import *
from nltk.util import ngrams

tokenizer = RegexpTokenizer('\w+')
stemmer = SnowballStemmer('english')

def grams(text, tokenizer=tokenizer, stemmer=stemmer):
 tokens = tokenizer.tokenize(text)
 tokens = list(map(stemmer.stem, tokens))

 unigrams = ngrams(tokens, 1)
 bigrams = ngrams(tokens, 2)

 return Counter(unigrams), Counter(bigrams)


def bigram_prob(unigrams, bigrams):
 probas = dict()

 for bigram in bigrams.elements():
  w1, _ = bigram
  probas[bigram] = bigrams[bigram] / unigrams[(w1,)]

 return probas