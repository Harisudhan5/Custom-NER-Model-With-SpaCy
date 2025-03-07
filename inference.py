import spacy

nlp_updated = spacy.load("ner_v3.0")
doc = nlp_updated("Flask is a lightweight and Json is standard")

print([(ent.text, ent.label_) for ent in doc.ents])