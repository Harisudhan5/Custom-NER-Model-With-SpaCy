
# Training Custom Named Entity Recognition Model with SpaCy

This repository contains a script for training any custom NER model easily using a SpaCy setup without worrying about more complicated configurations. The only requirement is a well annotated data for training.


### Clone the Repository
```bash
  git clone https://github.com/Harisudhan5/Train-Custom-NER-Model-With-SpaCy.git
```

### Setting up Dependencies

Install requirement file

```bash
pip install -r requirements.txt
```
Download a pre-trained model 

```bash
python -m spacy download en_core_web_lg
```

### Running a Pretrained NER Model

To run the pretrained NER model with its predefined entities, execute default.py to get the results

```bash
python default.py
```

### Training a Custom NER Model

1. Prepare the Dataset: Create a dataset and store it in data.py in the following format

```bash
training_data = [
    ["Python is one of the easiest languages to learn", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ['Support vector machines are powerful, but neural networks are more flexible.', {'entities': [[0, 22, 'ALGORITHM_MODEL'], [44, 59, 'ALGORITHM_MODEL']]}],
    ['I use Django for web development, and Flask for microservices.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [41, 46, 'FRAMEWORK_LIBRARY']]}]
]
```

2. Define New Labels: Modify train.py to define new entity labels in the list as per the prepared dataset.

3. Train the Model: Run the training script.

```bash
python train.py
```

The trained model will be saved as ner inside the Model Directory.

### Testing the Fine-Tuned Model

Once the model is trained, run inference.py to test the results

```bash 
python inference.py
```

### Important Note

By training a custom model via SpaCy, you may lose the pretrained entities such as GEO, ORG, etc. To retain previous knowledge, include those entities in your new dataset and make it diverse by adding multiple entity types in each sample.