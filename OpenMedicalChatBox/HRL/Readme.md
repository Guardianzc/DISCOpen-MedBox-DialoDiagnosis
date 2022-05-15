# HRL

```python
class HRL:
    def __init__(self, dataset_path, model_save_path, model_load_path, lr = 0.0005, groups = 4, cuda_idx = 0, train_mode = True, \
    max_turn = 10, reward_for_not_come_yet = 0, reward_for_success = 20, reward_for_fail = 0, reward_for_inform_right_symptom = 30, \
    reward_for_reach_max_turn = -100, reward_for_repeated_action = -4, epoch_number = 5000, epoch_size = 100, \
    experience_replay_size = 10000, batch_size = 100, discount_factor = 1, discount_factor_worker = 0.9, greedy = 0.1, reward_shaping = 1):
```

## Some important hyperparameters and their impact
|     Parameter     |       Impact   |    Default    | 
| :----------:      | :------------: | :-----------: | 
|     lr            |     The learning rate of the model, which determines the step size at each iteration while moving toward a minimum of a loss function.     |       0.0005  |
|     groups        |     The number of groups into which the disease is divided according to its symptom characteristics | 4(for mz-10), 3(for dxy), 2(for mz-4)....  |
|     cluster       |      When creating the dataset, we will divide diseases with a similarity higher than this value into a group. The larger the value, the fewer diseases in the group and the more similar in symptom characteristics | 0.5 (We advise based on some certain criteria (e.g. ICD-9-CM) when the number of diseases is large enough.)  
|     max_turn      |   The maximum number of symptoms that the agent can request | 10|
|     max_turn_workers | The maximum number of symptoms that the worker can request. The higher the value, the more times the same worker can request if the correct symptom is not obtained. You can tune it on "self.subtask_turn" in "agent/agent_hrl_joint2.py". | 2
|   experience_replay_size |  The number of data stored in the experience playback pool. The higher the value, the more data is used to update the model, and the slower the update speed is | 10000 (This value should be at least 1.5 times larger than the number of data used for training) |
| discount_factor_(worker) | The reward discount for master and worker, explains how the model evaluates future rewards. The higher the value, the higher the weight of future rewards in the policy | 1, 0.9|
| greedy | The probability that the model randomly chooses a action during training. The higher the value, the more likely the model is to randomly pick a action rather than picking the one with the highest expected return | 0.1|
| reward_shaping | The parameter for reward shaping techniques. The higher the value, the higher the model's reward for correctly predicting symptoms, and the lower for penalizing the mispredicting.| 1 |
