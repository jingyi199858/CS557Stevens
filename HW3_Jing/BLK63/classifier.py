import random

import nltk
from nltk.corpus import senseval

instances = senseval.instances('hard.pos')
size = int(len(instances) * 0.1)
train_set, test_set = instances[size:], instances[:size]

"""for i in train_set:
    print(i.context)
    """


def features(instance):
    feat = dict()
    p = instance.position
    if p:
        feat['wp'] = instance.context[p - 1][0]
        feat['tp'] = instance.context[p - 1][1]
    else:  #
        feat['wp'] = (p, 'BOS')
        feat['tp'] = (p, 'BOS')
        feat['wf'] = instance.context[p + 1][0]
        feat['tf'] = instance.context[p + 1][1]
    return feat


featureset = [(features(i), i.senses[0]) for i in instances if len(i.senses) == 1]

print(featureset)
random.shuffle(featureset)

train, dev, test = featureset[500:], featureset[:250], featureset[250:500]
classifier = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(classifier, train))
print(nltk.classify.accuracy(classifier, test))
