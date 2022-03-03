
import pickle
f0 = '.\\Fudan-Medical-Dialogue2.0\synthetic_dataset\\'
goal_set = pickle.load(file=open( f0 + '/goal_set.p', "rb"))
goal_test_set = pickle.load(file=open( f0 + '/goal_set.p', "rb"))
slot_set = pickle.load(file=open( f0 +  '//slot_set.p', "rb"))
disease_symptom = pickle.load(file=open(f0 +  '/disease_symptom.p', "rb"))

goal_train_set = goal_set['train'] + goal_test_set['test'] #+ goal_set['dev']  

imp_count = 0
all_count = 0
for info in goal_train_set:
    goal = info['goal']
    disease_tag = info['disease_tag']
    for symptom, state in goal['explicit_inform_slots'].items():
        all_count += 1
    for symptom, state in goal['implicit_inform_slots'].items():
        imp_count  += 1
        all_count += 1

imp_count /= len(goal_train_set)
all_count /= len(goal_train_set)
pass
