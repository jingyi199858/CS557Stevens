from nltk import *

def bi_grams(words):
    grams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return grams

def unsmoothed_unigrams(words):
    dist_words = FreqDist(words)
    pgrams = {}
    for gram in set(words):
        count_word = dist_words[gram]
        pgram = count_word / len(words)
        pgrams[gram] = pgram
    return pgrams

def unsmoothed_bigrams(words):
    grams = bi_grams(words)
    dist_words = FreqDist(words)
    dist_grams = FreqDist(grams)
    pgrams = {}
    for iGram in range(len(grams)):
        gram = grams[iGram]
        count_gram = dist_grams[gram]
        count_word_0 = dist_words[gram[0]]
        pgram = count_gram / count_word_0
        pgrams[gram] = pgram
    return pgrams

print(unsmoothed_unigrams("Good"))
print(unsmoothed_bigrams("use this"))