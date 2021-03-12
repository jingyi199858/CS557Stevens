import nltk
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
a = tag_fd.most_common()
#tag_fd.plot(cumulative=True)

total = len(brown_news_tagged)
print(total)
print(a)
for i in a:
    if i[0] == 'ADJ':
        print(i[1]/total)
    if i[0] == 'ADP':
        print(i[1]/total)
    if i[0] == 'ADV':
        print(i[1]/total)
    if i[0] == 'CONJ':
        print(i[1]/total)
    if i[0] == 'DET':
        print(i[1]/total)