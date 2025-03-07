import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from data import training_data
from spacy.lookups import Lookups
import random

new_labels = [ 
"PROGRAMMING_LANGUAGE",
"FRAMEWORK_LIBRARY",
"HARDWARE",
"ALGORITHM_MODEL",
"PROTOCOL",
"FILE_FORMAT",
"CYBERSECURITY_TERM",
]

epoch = 30

train_data = training_data
random.shuffle(train_data)

nlp = spacy.load("en_core_web_lg")

print(nlp.pipe_names)

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for data_sample, annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']


with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    epochs = epoch
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size = 128)
        for batch in batches:
            examples = []
            for text, annotations in  batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop = 0.15, losses = losses)
        print(f'Epoch : {epoch + 1}, Loss : {losses}')

nlp.to_disk('Model/ner')

print("Model is trained and saved to Model directory!!!")