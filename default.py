import spacy

texts = ['crypto advisors from England','latest trends in AI domain', 'UPI updates 2025']

nlp = spacy.load('en_core_web_lg')


docs = [nlp(text) for text in texts]

for doc in docs:
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    print(entities)


    