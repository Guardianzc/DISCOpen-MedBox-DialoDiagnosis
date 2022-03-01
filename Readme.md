# OpenMedicalChatbox

An Open-Source Package for Automatic Disease Diagnosis.

## Overview

Due to the lack of open source for existing RL-base automated diagnosis methods. It's hard to make a comparison for different methods. OpenMedicalChatbox integrates several current diagnostic methods and datasets. The documentation and tutorial of OpenMedicalChatbox are available at

## Dataset

At [here](./Data/Readme.md), we show all the mentioned datasets in existing medical methods, including MZ-4, Dxy, MZ-10 and a simulated dataset based on [Symcat](http://www.symcat.com/). In **goal.set** in their folders, explicit symptoms, implicit symptoms and diagnosis given by doctors are recorded for each sample. Also, we provide the corresponding tools to extend them for each methods. For further information, you can refer to the paper.

Here is the overview of datasets.

|     Name     | # of user goal | # of diseases | Ave. # of im. sym | # of sym. |
| :----------: | :------------: | :-----------: | :---------------: | :-------: |
|     MZ-4     |     1,733      |       4       |       5.46        |    230    |
|    MZ-10     |     3,745      |      10       |       5.28        |    318    |
|     Dxy      |      527       |       5       |       1.67        |    41     |
| SymCat-SD-90 |     30,000     |      90       |       2.60        |    266    |

 

## Methods

Besides, we reproduce several mainstream models for comparison.

1. **[Flat-DQN](http://www.aclweb.org/anthology/P18-2033)**: This is the baseline DQN agent, which has one layer policy and an action space including both symptoms and diseases. 
2. **[HRL-pretrained](https://ojs.aaai.org/index.php/AAAI/article/view/11902)**: This is a hierarchical model. The low level policy is pre-trained first and then the high level policy is trained. Besides, there is no disease classifier and the diagnosis is made by workers. 
3. **[REFUEL](https://proceedings.neurips.cc/paper/2018/hash/b5a1d925221b37e2e399f7b319038ba0-Abstract.html)**: This is a reinforcement learning method with reward shaping and feature rebuilding. It uses a branch to reconstruct the symptom vector to guide the policy gradient. 
4. **[KR-DS](https://ojs.aaai.org/index.php/AAAI/article/view/4722)**: This is an improved method based on Flat-DQN. It integrates a relational refinement branch and a knowledge-routed graph to strengthen the relationship between disease and symptoms. Here we adjust the code from [fantasySE](https://github.com/fantasySE/Dialogue-System-for-Automatic-Diagnosis).
5. **[GAMP](https://ojs.aaai.org/index.php/AAAI/article/view/5456)**: This is a GAN-based policy gradient network. It uses the GAN network to avoid generating randomized trials of symptom, and add mutual information to encourage the model to select the most discriminative symptoms.
6. **[HRL](https://arxiv.org/abs/2004.14254)**: This is a new hierarchical policy we purposed for diagnosis. The high level policy consists of a master model that is responsible for triggering a low level model, the low level policy consists of several symptom checkers and a disease classifier. Also, we try not to divide symptoms into different group (Denoted as **HRL (w/o grouped)**) to demonstrate the strength of two-level structure and remove the separate disease discriminator (Denoted as **HRL (w/o discriminator)**) to show the effect of disease grouping in symptom information extraction.



## QuickStart

1. Redirect the parameter **file0**  to the dataset needed. Note that if you use the KR-DS model, please redirect to "dataset_dxy" folder, and HRL dataset use the "HRL" folder.
2. Tune the parameter as you need.
3. Run the file or use the code below

DQN

```python
python ./HRL/src/run/run.py --agent_id agentdqn
```

HRL

```python
python ./HRL/src/run/run.py --agent_id agenthrljoint2
```

KR-DS

```python
python ./KR-DS/train.py 
```

REFUEL

```python
python ./REFUEL/train.py 
```

GAMP

```python
python ./GAMP/run.py 
```



## Experiment

We show the accuracy for disease diagnosis (**Acc.**), recall for symptom recovery (**M.R.**) and the average turns in interaction (**Avg. T**).

- In real world dataset

|                         |       |  Dxy  |       |       | MZ-4  |       |       | MZ-10 |       |
| :---------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model                   | Acc.  | M.R.  | Avg.T | Acc.  | M.R.  | Avg.T | Acc.  | M.R.  | Avg.T |
| Flat-DQN                | 0.731 | 0.110 | 1.96  | 0.681 | 0.062 | 1.27  | 0.408 | 0.047 | 9.75  |
| KR-DS                   | 0.740 | 0.399 | 5.65  | 0.678 | 0.177 | 4.61  | 0.485 | 0.279 | 5.95  |
| REFUEL                  | 0.721 | 0.186 | 3.11  | 0.716 | 0.215 | 5.01  | 0.505 | 0.262 | 5.50  |
| GAMP                    | 0.731 | 0.268 | 2.84  | 0.644 | 0.107 | 2.93  | 0.500 | 0.067 | 1.78  |
| Classifier Lower Bound  | 0.682 |  --   |  --   | 0.671 |  --   |  --   | 0.532 |  --   |  --   |
| HRL (w/o grouped)       | 0.731 | 0.297 | 6.61  | 0.689 | 0.004 | 2.25  | 0.540 | 0.114 | 4.59  |
| HRL (w/o discriminator) |  --   | 0.512 | 8.42  |  --   | 0.233 | 5.71  |  --   | 0.330 | 8.75  |
| HRL                     | 0.779 | 0.424 | 8.61  | 0.735 | 0.229 | 5.08  | 0.556 | 0.295 | 6.99  |
| Classifier Upper Bound  | 0.846 |  --   |  --   | 0.755 |  --   |  --   | 0.612 |  --   |  --   |

-  In synthetic dataset

| Model                  | Acc.  | M.R.  | Avg.T |
| ---------------------- | ----- | ----- | ----- |
| Flat-DQN               | 0.343 | 0.023 | 1.23  |
| KR-DS                  | 0.357 | 0.388 | 6.24  |
| REFUEL                 | 0.347 | 0.161 | 4.56  |
| GAMP                   | 0.267 | 0.077 | 1.36  |
| Classifier Lower Bound | 0.308 | --    | --    |
| HRL-pretrained         | 0.452 | --    | 3.42  |
| HRL                    | 0.504 | 0.495 | 6.48  |
| Classifier Upper Bound | 0.781 | --    | --    |



## Reference

- [Task-oriented Dialogue System for Automatic Diagnosis](https://aclanthology.org/P18-2033.pdf)
- [Context-Aware Symptom Checking for Disease Diagnosis Using Hierarchical Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11902)
- [REFUEL: Exploring Sparse Features in Deep Reinforcement Learning for Fast Disease Diagnosis](https://proceedings.neurips.cc/paper/2018/hash/b5a1d925221b37e2e399f7b319038ba0-Abstract.html)
- [End-to-End Knowledge-Routed Relational Dialogue System for Automatic Diagnosis](https://ojs.aaai.org/index.php/AAAI/article/view/4722)
- [Generative Adversarial Regularized Mutual Information Policy Gradient Framework for Automatic Diagnosis](https://ojs.aaai.org/index.php/AAAI/article/view/5456)
- [Task-oriented Dialogue System for Automatic Disease Diagnosis via Hierarchical Reinforcement Learning](https://arxiv.org/abs/2004.14254)



## Citation

Please cite our paper if you use toolkit

```
@article{liao2020task,
  title={Task-oriented dialogue system for automatic disease diagnosis via hierarchical reinforcement learning},
  author={Liao, Kangenbei and Liu, Qianlong and Wei, Zhongyu and Peng, Baolin and Chen, Qin and Sun, Weijian and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2004.14254},
  year={2020}
}
```

