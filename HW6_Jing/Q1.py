import nltk


"""
Ambiguous words: refuse, permit, book, cook.
Same tag: on, in
"""
text1 = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
text2 = nltk.word_tokenize("The book is not on the table, it is in my backpack")
text3 = nltk.word_tokenize("Book the cooks who cook the books.")
a = nltk.pos_tag(text1)
b = nltk.pos_tag(text2)
c = nltk.pos_tag(text3)

print(a)
print(b)
print(c)

"""
Report:
"IN" tag refers to Preposition or subordinating conjunction. In the sentence:
"The book is not on the table, it is in my backpack", both "on" and "in" are tagged 
as "IN". Since "on" and "in" serves as preposition; they link nouns in this sentence.

The sentence "Book the cooks who cook the books." is basically the same as the first sentence
"They refuse to permit us to obtain the refuse permit"; they both have two ambiguous words that 
each serves different part of speech in the sentence. 
"""