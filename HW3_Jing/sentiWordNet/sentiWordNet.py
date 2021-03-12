import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]


ps = PorterStemmer()
words_data = ['this','movie','is','wonderful']
words_data1 = ['this','movie','is','awful']
words_data2 = ['this','movie','is','not', 'attracting']
words_data3 = [ 'Excellent', 'noise canceling', 'and', 'sound', 'quality,', 'along', 'with', 'a', 'very', 'good', 'mobile', 'application']

pos_val = nltk.pos_tag(words_data)
senti_val = [get_sentiment(x,y) for (x,y) in pos_val]

pos_val1 = nltk.pos_tag(words_data1)
senti_val1 = [get_sentiment(x,y) for (x,y) in pos_val1]

pos_val2 = nltk.pos_tag(words_data2)
senti_val2 = [get_sentiment(x,y) for (x,y) in pos_val2]

pos_val3 = nltk.pos_tag(words_data3)
senti_val3 = [get_sentiment(x,y) for (x,y) in pos_val3]


print(senti_val)
print(senti_val1)
print(senti_val2)
print(senti_val3)