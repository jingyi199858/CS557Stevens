import nltk

text1 = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
text2 = nltk.word_tokenize("Left brain has nothing left")
text3 = nltk.word_tokenize("I can bring a garbage can")
text4 = nltk.word_tokenize("They wave their hands in that giant wave")
a = nltk.pos_tag(text1)
b = nltk.pos_tag(text2)
c = nltk.pos_tag(text3)
d = nltk.pos_tag(text4)

print(a)
print(b)
print(c)
print(d)
