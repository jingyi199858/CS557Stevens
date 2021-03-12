from nltk.corpus import words
from random import sample
#generate random sentence
def geneSent(n):
    n = 100
    rand_words = ' '.join(sample(words.words(), n))
    return rand_words

sent = geneSent(100)
print(sent)