# OpenMedicalChatbox

An Open-Source Package for Automatic Disease Diagnosis.

## Overview

Due to the lack of open source for existing RL-base automated diagnosis methods. It's hard to make a comparison for different methods. OpenMedicalChatbox integrates several current diagnostic methods and datasets.

## Dataset

At [here](./Data/Readme.md), we show all the mentioned datasets in existing medical methods, including MZ-4, Dxy, MZ-10 and a simulated dataset based on [Symcat](http://www.symcat.com/). In **goal.set** in their folders, explicit symptoms, implicit symptoms and diagnosis given by doctors are recorded for each sample. Also, we provide the corresponding tools to extend them for each methods. 

Here is the overview of datasets.

|     Name     | # of user goal | # of diseases | Ave. # of im. sym | # of sym. |
| :----------: | :------------: | :-----------: | :---------------: | :-------: |
|     MZ-4     |     1,733      |       4       |       5.46        |    230    |
|    MZ-10     |     3,745      |      10       |       5.28        |    318    |
|     Dxy      |      527       |       5       |       1.67        |    41     |
| SymCat-SD-90 |     30,000     |      90       |       2.60        |    266    |

 

## Methods

Besides, we reproduce several mainstream models for comparison. For further information, you can refer to the [paper](./paper/).

1. **[Flat-DQN](http://www.aclweb.org/anthology/P18-2033)**: This is the baseline DQN agent, which has one layer policy and an action space including both symptoms and diseases. 
2. **[REFUEL](https://proceedings.neurips.cc/paper/2018/hash/b5a1d925221b37e2e399f7b319038ba0-Abstract.html)**: This is a reinforcement learning method with reward shaping and feature rebuilding. It uses a branch to reconstruct the symptom vector to guide the policy gradient. 
3. **[KR-DS](https://ojs.aaai.org/index.php/AAAI/article/view/4722)**: This is an improved method based on Flat-DQN. It integrates a relational refinement branch and a knowledge-routed graph to strengthen the relationship between disease and symptoms. Here we adjust the code from [fantasySE](https://github.com/fantasySE/Dialogue-System-for-Automatic-Diagnosis).
4. **[GAMP](https://ojs.aaai.org/index.php/AAAI/article/view/5456)**: This is a GAN-based policy gradient network. It uses the GAN network to avoid generating randomized trials of symptom, and add mutual information to encourage the model to select the most discriminative symptoms.
5. **[HRL](https://arxiv.org/abs/2004.14254)**: This is a new hierarchical policy we purposed for diagnosis. The high level policy consists of a master model that is responsible for triggering a low level model, the low level policy consists of several symptom checkers and a disease classifier. Also, we try not to divide symptoms into different group (Denoted as **HRL (w/o grouped)**) to demonstrate the strength of two-level structure and remove the separate disease discriminator (Denoted as **HRL (w/o discriminator)**) to show the effect of disease grouping in symptom information extraction.



## Installation

1. Install the packages
```python 
pip install OpenMedicalChatBox
```
   or Cloning this repo

```python
git clone https://github.com/Guardianzc/OpenMedicalChatBox.git
cd OpenMedicalChatBox
python setup.py install
```

After installation, you can choose the dataset and method, then try running  ` demo.py` to check if OpenMedicalChatBox works well

```
python demo.py
```



2. Redirect the parameter **file0**  to the dataset needed. Note that if you use the KR-DS model, please redirect to "dataset_dxy" folder, and HRL dataset use the "HRL" folder.
3. Tune the parameter as you need.
4. Run the file or use the code below



## Examples

The following code shows how to use OpenMedicalChatBox to apply different diagnosis method on datasets.

```python
import OpenMedicalChatBox as OMCB
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

HRL_test = OMCB.HRL(dataset_path = './/OpenMedicalChatBox/Data/mz4/HRL', model_save_path = './simulate//', groups = 2, model_load_path = './simulate/DQN/checkpoint/0411092858_MZ-10_agenthrljoint2_T20_ss100_lr0.0005_RFS20_RFF0_RFNCY0_RFIRS30_RFRA-4_RFRMT-100_gamma1_gammaW0.9_epsilon0.1_crs0_wfrs1_RID0/model_d10agenthrljoint2_s0.299_r-20.951_t9.5_mr0.007_mr2-0.004_e-0.pkl', cuda_idx = 1, train_mode = False)
HRL_test.run()


KRDS_test = OMCB.KRDS(dataset_path = './/OpenMedicalChatBox//Data//mz4//dataset_dxy//', model_save_path = './simulate//', model_load_path = None, cuda_idx = 1, warm_start = 1, train_mode = True)
KRDS_test = OMCB.KRDS(dataset_path = './/OpenMedicalChatBox//Data//mz4//dataset_dxy//', model_save_path = './simulate//', model_load_path = './simulate/test_2_2_0.403_1.977_0.060.pth.tar', cuda_idx = 1, warm_start = 1, train_mode = False)
KRDS_test.run()


Flat_DQN_test = OMCB.Flat_DQN(dataset_path = './/OpenMedicalChatBox//Data//mz4//', model_save_path = './simulate//',  model_load_path = '/remote-home/czhong/RL/DISCOpen-MedBox/simulate/DQN/checkpoint/0411114102_MZ-10_agentdqn_T20_ss100_lr0.0005_RFS20_RFF0_RFNCY0_RFIRS6_RFRA-4_RFRMT-100_gamma1_gammaW0.9_epsilon0.1_crs0_wfrs1_RID0/model_d10agentdqn_s0.299_r6.417_t2.5_mr0.024_mr2-0.014_e-2.pkl', cuda_idx = 1, warm_start=True ,train_mode = True)
Flat_DQN_test.run()


GAMP_test = OMCB.GAMP(dataset_path = './/OpenMedicalChatBox//Data//mz4//', model_save_path = './simulate//', model_load_path = './simulate/0411125423/s0.612_obj2.652_t2.954_mr0.107_outs0.183_e-0', cuda_idx = 0, train_mode = True)
GAMP_test.run()

REFUEL_test = OMCB.REFUEL(dataset_path = './/OpenMedicalChatBox//Data//mz4//', model_save_path = './simulate//', model_load_path = './simulate/0411132328/s9.043_obj-16.433_t1.0_mr0.0_outs0.0_e-1.pkl', cuda_idx = 0, train_mode = True)
REFUEL_test.run()

```

 

The detail experimental parameters are shown in [here](./OpenMedicalChatBox/Readme.md).



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



## Reference

- [Task-oriented Dialogue System for Automatic Diagnosis](https://aclanthology.org/P18-2033.pdf)
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

