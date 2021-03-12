import nltk
from nltk import word_tokenize

"""
Ambiguous words: book, cook, show.
"""
first = "Book the cooks who cook the books."
second = "Game show contestant told he's not allowed to show the studio audience's mindless shouting"
First = word_tokenize(first)
print(nltk.pos_tag(First))
Second = word_tokenize(second)
print(nltk.pos_tag(Second))

"""
Ambiguous word: juvenile.
"""
exampleOne = word_tokenize("he acted very juvenile")
exampleTwo = word_tokenize("Juvenile Court to Try Shooting Defendant")
print(nltk.pos_tag(exampleOne))
print(nltk.pos_tag(exampleTwo))

"""
Report:
The first sentence is the same from question one. Each ambiguous word serves different part of 
speech in the sentence.
Next sentence represents a typical example of ambiguous word. "show" acts as both noun and verb in
the sentence.
Last example can have multiple meanings:
Actual meaning:A person who is being accused of shooting someone will appear in court. (court trial)
Alternative interpretation:The court will attempt at shooting (killing) the person who is being accused.
The tagging result leaning to alternative interpretation, since it's more related to the structure of 
that sentence. Therefore, the problem is that the taggers tend to define a sentence more based on 
structure. 

"""