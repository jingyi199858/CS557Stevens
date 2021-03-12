from functools import reduce

import numpy as np

documents = [
    ('pos', 3 * ["good"] + 3 * ["great"]),
    ('pos',  ["poor"] + 2 * ["great"]),
    ('neg', ["good"] + 3 * ["poor"]),
    ('neg', ["good"] + 5 * ["poor"] + 2 * ["great"]),
    ('neg', 2 * ["poor"])
]

doc = ['A', 'good', 'good', 'plot', 'and', 'great', 'characters', 'but', 'poor', 'acting']

classes = set([x[0] for x in documents])

for binarize in [True, False]:
    lp, ll, V = train_naive_bayes(documents, classes, binarize)
    print(test_naive_bayes(doc, lp, ll, classes, V))