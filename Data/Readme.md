# Dataset

This file includes four datasets and their tools to expand.

## Introduction

**[MZ-4](http://www.aclweb.org/anthology/P18-2033)**: This is the first dataset collected from real environment for the evaluation of task-oriented dataset. It includes 4 diseases, 230 symptoms and 1,733 user goals. Each user record consists of the self-report provided by the user and conversation text between the patient and a doctor. Symptoms extracted from self-report are treated as explicit symptoms and the ones extracted from conversation are implicit symptoms.

**[MZ-10](https://arxiv.org/abs/2004.14254)**: It is expanded from MZ-4 to include 10 diseases, consisting of typical diseases of digestive system, respiratory system and endocrine system. The raw data is collected from the pediatric department on a [Chinese online healthcare community](http://muzhi.baidu.com). We hire experts with medical background to identify symptom expressions and label them with three tags (True, False or UNK) to indicate whether the user suffers this symptom. After that, experts manually link each symptom expression to a concept on [SNOMED CT](https://www.snomed.org/snomed-ct).

**[Dxy](https://ojs.aaai.org/index.php/AAAI/article/view/4722):** A Dialogue Medical dataset contains data from a [Chinese online healthcare website](https://dxy.com/). This dataset contains 527 user goals, including 5 diseases and 41 specific symptoms. 

**[SymCat-SD-90](](https://arxiv.org/abs/2004.14254)):** It is constructed based on symptom-disease database called [SymCat](www.symcat.com). There are 801 diseases in the database and we classify them into 21 departments (groups) according to International Classification of Diseases [ICD-10-CM](https://www.cdc.gov/nchs/icd/). We choose 9 representative departments from the database, each department contains top 10 diseases according to the occurrence rate in the Centers for Disease Control and Prevention (CDC) database.

## Overview

|     Name     | # of user goal | # of diseases | Ave. # of im. sym | # of sym. |
| :----------: | :------------: | :-----------: | :---------------: | :-------: |
|     MZ-4     |     1,733      |       4       |       5.46        |    230    |
|    MZ-10     |     3,745      |      10       |       5.28        |    318    |
|     Dxy      |      527       |       5       |       1.67        |    41     |
| SymCat-SD-90 |     30,000     |      90       |       2.60        |    266    |

 

## Quick Start

Loading this dataset using the following command in Python:

```python
import pickle
data_set = pickle.load(open(file_name, 'rb'))
```

 

## Important Files

In each dataset,

- *similarity.py:* Tools for evaluate the similarity between diseases.
- *action_set.p*: the types of action pre-defined for this dataset.
- *disease_set.p:* Diseases included in this dataset.
- *slot_set.p:* Symptoms included in this dataset.
- *goal.set.p & goal_test_set.p:* It contains training set and testing set, which can be visited with goal_set["train"] and goal_set["test"] 
- *disease_symptom.p*: the number of symptoms occurred in each disease.

**Tools:**

- *count.py:*  Tools for generating overview for each dataset.
- *HRL_generate.py:* Tools for dividing diseases into different groups based on similarity, which forming HRL folders.
- *dataset_dxy/data_process.py:* Generate a probability matrix between diseases and symptoms, which will be applied to the KR-DS method 
