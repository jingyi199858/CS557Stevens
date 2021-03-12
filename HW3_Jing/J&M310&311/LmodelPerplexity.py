import collections
import nltk

def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word] / N
    return model

#compute perplexity
def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity

corpus = """
Nine states are seeing double-digit increases in the number of patients hospitalized because of coronavirus complications, according to data tracked by The Washington Post.
Wyoming and Montana reported the highest inpatient rolling average with a 40 percent and 27 percent change in the averages, respectively, according to the data."""
tokens = nltk.word_tokenize(corpus)
model = unigram(tokens)
print(model)
set1 = "coronavirus"
set2 = "Wyoming and Montana"
print(perplexity(set1, model))
