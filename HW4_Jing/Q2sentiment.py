import pandas as pd
from sklearn.linear_model import LogisticRegression
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
df = pd.read_csv('Reviews.csv')
X = df['Text']
Y = df['Score']
X_train = X[0:1000]
Y_train = Y[0:1000]
with open('positive-words.txt') as f:
    positive=f.readlines()
with open('negative-words.txt') as l:
    negative = l.readlines()
with open('negator-words.txt') as j:
    negator = j.readlines()
for i in range(len(positive)):
    positive[i] = positive[i].replace("\n", "")
for i in range(len(negative)):
    negative[i] = negative[i].replace("\n", "")
for i in range(len(negator)):
    negator[i] = negator[i].replace("\n", "")
fa_words = []
ne_words = []
na_words = []
count = 0
count_n = 0
count_na = 0
test_fa_words = []
test_ne_words = []
test_na_words = []
test_count = 0
test_count_n = 0
test_count_na = 0
for i in X_train:
    i1 = re.sub('[+\.\!\/_,$%^*(+\")]+|[+——()?:【】“”！，。？、~@#￥%……&*（）]+', "", i)
    a= i1.split()
    for l in range(len(a)):
        if a[l] in positive:
            count = count + 1
        if a[l] in negative:
            count_n = count_n +1
        if a[l] in negator:
            count_na = count_na +1
    fa_words.append(count)
    ne_words.append(count_n)
    na_words.append(count_na)
    count = 0
    count_n = 0
    count_na = 0

x_df = np.column_stack((fa_words, na_words, ne_words))

clf = LogisticRegression()
clf.fit(x_df, Y_train)
pred = clf.predict(x_df)

pred1 = []
for i in range(len(pred)):
    if pred[i] == 1:
        pred1.append('very unfavorable')
    if pred[i] == 2:
        pred1.append('unfavorable')
    if pred[i] == 3:
        pred1.append('neutral')
    if pred[i] == 4:
        pred1.append('favorable')
    if pred[i] == 5:
        pred1.append('very favorable')
y_train = []
for i in range(len(Y_train)):
    if Y_train[i] == 1:
        y_train.append( 'very unfavorable')
    if Y_train[i] == 2:
        y_train.append('unfavorable')
    if Y_train[i] == 3:
        y_train.append('neutral')
    if Y_train[i] == 4:
        y_train.append('favorable')
    if Y_train[i] == 5:
        y_train.append('very favorable')
confusion = confusion_matrix(y_train, pred1 )

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm = confusion, normalize = False, target_names =['very_unfavorable', 'unfavorable', 'neutral', 'favorable', 'very_favorable'],title = "Confusion matrix" )
