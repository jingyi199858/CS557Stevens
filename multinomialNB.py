# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:04:14 2019
Text classification & NLP using Multinominal Naive Bayes
"""
# Step 1 : Import libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer

# word stemmer
stemmer = LancasterStemmer()

# Step 2 : Provide training data
# 3 classes of training data
training_data = []
# greeting class
training_data.append({"class": "greeting", "sentence": "how are you?"})
training_data.append({"class": "greeting", "sentence": "how is your day?"})
training_data.append({"class": "greeting", "sentence": "good day"})
training_data.append({"class": "greeting", "sentence": "how is it going today?"})
# goodbye class
training_data.append({"class": "goodbye", "sentence": "have a nice day"})
training_data.append({"class": "goodbye", "sentence": "see you later"})
training_data.append({"class": "goodbye", "sentence": "see ya"})
training_data.append({"class": "goodbye", "sentence": "talk to you soon"})
# sandwich calss
training_data.append({"class": "sandwich", "sentence": "make me a sandwich"})
training_data.append({"class": "sandwich", "sentence": "can you make a sandwich?"})
training_data.append({"class": "sandwich", "sentence": "having a sandwich today?"})
training_data.append({"class": "sandwich", "sentence": "what's for lunch?"})
print("%s sentences of training data" % len(training_data))

# Step 3 : Organize data in structures
# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s"]:
            # stem & lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

                # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])

# we now have each stemmed word and the number of occurances of the word in our training corpus(the word's commonality)
print("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
print("Class words: %s" % class_words)


# Step 4 : Code algorithm
# calculate a score for a given class
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            '''# treat each word with same weight
            score += 1
            '''
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                '''print("match: %s" % stemmer.stem(word.lower()))'''
                print("match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score


# we can now calculate score for new sentence
sentence = "A good, good plot and great characters, but poor acting."
# now we can find the class with the highest score
for c in class_words.keys():
    print("Class: %s, Score: %s\n" % (c, calculate_class_score(sentence, c)))


# Step 5 : Abstract algorithm
# return the class with highest score for sentence
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in classes:
        # calculate score of sentence for each class
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest sore
        if score > high_score:
            high_score = score
            high_class = c

    return high_score, high_class