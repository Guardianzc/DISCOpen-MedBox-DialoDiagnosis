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
|     lr            |     The learning rate of model, determines the step size at each iteration while moving toward a minimum of a loss function.     |       0.0005  |
|     groups        |     The number of groups into which the disease is divided according to its symptom characteristics | 4(for mz-10), 3(for dxy), 2(for mz-4)....  |
|     cluster       |      When creating the dataset, we will divide diseases with a similarity higher than this value into a group. The larger the value, the fewer diseases in the group and the more similar in symptom characteristics | 0.5 (We advice to based on some on certain criteria (e.g. ICD-9-CM) when the number of disease is large enough.)  
