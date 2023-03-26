# Helmet detection project

Final project of learning Data Science in SberUniversiry group DS14AN subgroup 5 - FrozenSet.

## Task

The main task of project is creation of model capable to detect the presence of construction helmet (hard top hat) onto the persons head in different locations. This can be useful, for example, inside industrial and construction facility for safety reasons.

## Proposed way

As the solution the using of **Convolution Neural Networks (CNN)** is proposed.
The main idea is training of the CNN with two selected classes:
- *helmet* for head with construction helmet
- *head* for head withot construction helmet

So it is supposed that the trained model will be able to predict presence of the helmet on the person's head in any image.

To solve the task, the following elements are necessary:

1. Dataset for training with two demanded classes annotations, because the supervised learning is proposed.
2. Convolution Neural Networks Model.
3. Interface for prediction routine.

## Dataset

### Description
Dataset - VOC2028 with marks: "person" for head and "helmet" for head with construction helmet.

[Link to the project](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)

[Download dataset from Google Drive](https://drive.google.com/file/d/1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX/view)

### Short description

SafetyHelmetWearing-Dataset(SHWD)

"SHWD provide the dataset used for both safety helmet wearing and human head detection. It includes 7581 images with 9044 human safety helmet wearing objects(positive) and 111514 normal head objects(not wearing or negative)."

In the above short description provided with the dataset the typo is revealed: the number of helmet class objects in annotations is 9058 and the number of head class objects is 111710.

### Train/validation split

To validation dataset allocated 20% of objects of each class (not number of files) with maintaining class ratio, i.e. numbers objects of class helmet are 1812 and 22349 of class head in totally 1570 files. Files for validation dataset are chosen randomly to collect demanded number of objects (obviously, other part of files used to train). The list of files names for validation/train dataset is saved in json format in files helmet_valid_filenames.json/helmet_train_filenames.json and presented in this project inside the folder VOC2028.

The way to read data (example for reading validation dataset files names):

```
with open(dataset_path + "helmet_valid_filenames.json", 'r') as f:
    helmet_valid = json.load(f)
```
### Bounding boxes data

All useful information (including bounding boxes data for each class and number of each object) obtained from xml annotations are collected in json format file named data.json and placed in folder VOC2028.

## Model