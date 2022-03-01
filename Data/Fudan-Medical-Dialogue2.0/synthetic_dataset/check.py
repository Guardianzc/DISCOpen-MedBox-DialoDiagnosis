import pickle
slot_set_all = list()
for i in ['1','4','5','6','7','12','13','14','19']:
    slot_set = pickle.load(file=open('./label' + i + '/slot_set.p', "rb"))
    slot_set_all = slot_set_all + list(slot_set.keys())
pass
slot_set_all = set(slot_set_all)
disease_symptom = pickle.load(file=open('./disease_symptom.p', "rb"))
print(1)