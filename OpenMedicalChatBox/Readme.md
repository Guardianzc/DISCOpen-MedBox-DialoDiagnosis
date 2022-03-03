# OpenMedicalChatbox

An Open-Source Package for Automatic Disease Diagnosis.

## Parameter Setting

### Flat-DQN

```python
class Flat_DQN:
    def __init__(self, dataset_path, model_save_path, model_load_path, lr = 0.0005, cuda_idx = 0, train_mode = True, max_turn = 10, reward_for_not_come_yet = 0, reward_for_success = 20, reward_for_fail = 0, reward_for_inform_right_symptom = 6, reward_for_reach_max_turn = -100, reward_for_repeated_action = -4, epoch_number = 5000, epoch_size = 100, experience_replay_size = 10000, batch_size = 100, discount_factor = 1, warm_start = False, warm_start_number = 30, greedy = 0.1):
```

### HRL

```python
class HRL:
    def __init__(self, dataset_path, model_save_path, model_load_path, lr = 0.0005, groups = 4, cuda_idx = 0, train_mode = True, max_turn = 10, reward_for_not_come_yet = 0, reward_for_success = 20, reward_for_fail = 0, reward_for_inform_right_symptom = 30, reward_for_reach_max_turn = -100, reward_for_repeated_action = -4, epoch_number = 5000, epoch_size = 100, experience_replay_size = 10000, batch_size = 100, discount_factor = 1, discount_factor_worker = 0.9, greedy = 0.1, reward_shaping = 1):
```

### GAMP

```PYTHON
class GAMP:
    def __init__(self, dataset_path, model_save_path, model_load_path, cuda_idx = 0, epoch_number = 1000, train_mode = True, max_turn = 10 ,batch_size = 64, lr = 0.0001):
```

### KRDS

```python
class KRDS:
    def __init__(self, dataset_path, model_save_path, model_load_path, cuda_idx, train_mode = True, greedy = 0.1, epoch_number = 5000, max_turn = 10, experience_replay_size = 10000, batch_size = 32, reward_for_not_come_yet = 0, reward_for_success = 8, reward_for_fail = 0, reward_for_inform_right_symptom = 6, reward_for_reach_max_turn = -100, reward_for_repeated_action = -4, lr = 0.01, discount_factor = 0.9, warm_start = True, warm_start_number = 5000):
```

### REFUEL

```python
class REFUEL:
    def __init__(self, dataset_path, model_save_path, model_load_path, cuda_idx, train_mode = True, epoch_number = 5000, batch_size = 64, max_turn = 10, reward_shaping = 0.25,  reward_for_success = 20, reward_for_fail = -1,  reward_for_reach_max_turn = -1, rebuild_factor = 10, entropy_factor = 0.007, discount_factor = 0.99, lr = 0.0001):
```



#### Training Setting

**dataset_path:** The path of dataset.

**model_save_path:** The path to save models.

**model_load_save:** The model to load models. ( When testing )

**lr:** Learning rate.

**cuda_idx:** The gpu index you use to train the model.

**train_mode:** Whether to train or test.

**max_turn:** The max turn of dialogue. (How many symptom you can request.)



#### Reward Setting

**reward_for_not_come_yet:** Reward for request the wrong symptom.

**reward_for_inform_right_symptom:** Reward for request the right symptom.

**reward_for_success:** Reward for inform the right disease.

**reward_for_fail:** Reward for inform the wrong disease.

**reward_for_reach_max_turn**: Reward when dialogue reach the max turn.

**reward_for_repeat_action**: Reward when dialogue reach the max turn.



#### Model Setting

**epoch_number:**  The number of epochs.

**epoch_size:** The number of size for each epoch.

**experience_replay_size:** The size of experience replay pool.  

**batch_size:** Batch size when training the model.

**discount_factor:** The discount factor in Reinforcement Learning

**warm_start:** Whether to warm start the model.

**warm_start_number:** The epoch number of warm start.

**greedy**:  The probability to execute the greedy exploration.



#### HRL

**groups**:  The number of groups you divide all diseases into. ( The number of workers )

**reward_shaping:** The reward shaping factor. ( 0 means no reward shaping )



#### REFUEL

**rebuild_factor:** The factor for rebuilding the symptom vector.

**entropy_factor:** The entropy factor for the probability of policy.



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

