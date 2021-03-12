import nltk

wsj = nltk.corpus.treebank.tagged_words()

cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)

word_tag_pairs = nltk.bigrams(nltk.corpus.treebank.tagged_words())

vbn_preceders = [(a[0],b[0]) for (a,b) in word_tag_pairs if b[1]=='VBN' and b[0] in list(cfd2['VBN'])]
print(vbn_preceders)