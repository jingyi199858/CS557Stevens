import gensim, logging, os
import nltk
import numpy as np

from confusionPlot import plot_confusion_matrix

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# the corpus is brown from nltk
corpus = nltk.corpus.brown.sents()

name = 'brown_skipgram.model'
if os.path.exists(name):
    # load the file if it has already been trained, to save repeating the slow training step below
    model = gensim.models.Word2Vec.load(name)
else:
    # can take a few minutes, grab a cuppa
    model = gensim.models.Word2Vec(corpus, size=100, min_count=5, workers=2, iter=50)
    model.save(name)

words = "John Mary like to watch football games movies".split()
for w1 in words:
    for w2 in words:
        print(w1, w2, model.similarity(w1, w2))

M = np.zeros((len(words), len(words)))
for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        M[i,j] = model.similarity(w1, w2)
        M[i,j] = round(M[i,j], 4)

plot_confusion_matrix(cm = M, normalize = False, target_names =words,title = "Confusion matrix" )