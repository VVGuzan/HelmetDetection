# Helmet detection project

Final project of learning Data Science in SberUniversiry group DS14AN subgroup 5 - FrozenSet. 

Here ***only my personal part of project*** is presented (SSDLite and some additional model not presented in final group project). The final group project can be found at this [link](https://github.com/Danielnex7/frozenset).

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

## Models

### SSD Lite 

As the basic simple model for the detection task the SSD Lite model was chosen (the additional models in the project were different versions of YOLO: from v5 up to last v8). The SSD Lite model is a "one-shot" model as YOLO but simpler. One can foud additional info at this [link](https://pytorch.org/vision/main/models/ssdlite.html), containing links to fundamental works.

## Training

### General review of training process

The model was trained using the train part of dataset (defined early). The quality of model for each epoch was checked by calclulating losses (classification loss - cross-entropy and box loss - regression for box korners coordinates) and some metrics (map50, precision, recall, f1-score).

The losses was calculated for train and validation part by buildin loss calculation proceedure provided by standard train proceedure (train mode of model).

To calculate metrics on validation part of dataset additional library called `torchmetrics` was used.

Whole proceedure of train (with metrics calculation) is presented in notebook `SSDLite_trai_val_metrics.ipynb`.

### Trained models 

Three different models of SSD Lite (depending of freezed/trainable layers) was trained:
1. Fully trainable model
2. Only "head" layers are trainable
3. Only deep backbone layers (Mobile Net) freezed

### Choose the best model

For each case (1 - 3) the best model was chosen by metrics values (map50).

## Resolved difficulties

### At training stage

#### Data transformation

#### Losses calculation

### At validation stage

#### Metrics calculation