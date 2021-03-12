from nltk.corpus import wordnet as wn

print(wn.synsets('body'))
print(wn.synset('body.n.01').member_meronyms())
print(wn.synset('body.n.01').part_meronyms())
print(wn.synset('body.n.01').substance_meronyms())
print(wn.synset('body.n.01').member_holonyms())
print(wn.synset('body.n.01').part_holonyms())
print(wn.synset('body.n.01').substance_holonyms())


def supergloss(s):
    description = s.definition()
    hypernyms = s.hypernyms()
    hypernyms_definition = ""
    for hypernym in hypernyms:
        hypernyms_definition += hypernym.definition()

    hyponyms = s.hyponyms()
    hyponyms_definition = ""
    for hyponym in hyponyms:
        hyponyms_definition += hyponym.definition()

    return description + "hypernyms: " + hypernyms_definition + "hyponyms: " + hyponyms_definition


print(supergloss(wn.synset('car.n.01')))


def polysemy_calc(type):
    total = 0
    polysemy_count = 0

    for synset in wn.all_synsets():
        if synset.pos() == type:
            for lemma in synset.lemmas():
                total += 1
                count = 0
                count += len(wn.synsets(lemma.name(), synset.pos()))
                polysemy_count += count

    return polysemy_count * 1.0 / total


print("noun:" + str(polysemy_calc('n')))
print("verb:" + str(polysemy_calc('v')))
print("adj.:" + str(polysemy_calc('a')))
