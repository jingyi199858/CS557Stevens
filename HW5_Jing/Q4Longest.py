from nltk.tokenize import sent_tokenize
from nltk.book import *
from nltk.tokenize.treebank import TreebankWordDetokenizer

booklist = [text1,text2,text3,text4,text5,text6,text7,text8,text9]
longest = []

for i in booklist:
    text_to = TreebankWordDetokenizer().detokenize(i)
    longest.append(max(sent_tokenize(text_to), key=len))

for i in longest:
    print(len(i))

print(len(max(longest, key=len)), max(longest, key=len))