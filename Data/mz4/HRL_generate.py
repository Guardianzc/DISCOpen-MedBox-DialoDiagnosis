import pickle

goal_set = pickle.load(file=open('.//goal_set.p', "rb"))
slot_set = pickle.load(file=open('.//slot_set.p', "rb"))
goal_test_set = pickle.load(file=open('.//goal_test_set.p', "rb"))
output_files = './HRL/'
slot_set_key = dict()
disease_set = list()
disease_set_dict = dict()
disease_symptom = {'label1':dict(), 'label2':dict()}
slot_set_dict = {'label1':dict(), 'label2':dict()}
goal_set_HRL = dict()
goal_set_HRL_total = dict()

for i in ['label1','label2','label3','label4']:
    goal_set_HRL[i] = dict()
    for j in ['train','test','validate']:
        goal_set_HRL[i][j] = list()
        goal_set_HRL_total[j] = list()

count = 0
classify = dict()
for i in ['小儿腹泻', '小儿消化不良']:
    classify[i] = 'label1'
    disease_symptom['label1'][i] = dict()
    disease_symptom['label1'][i]['index'] = count
    disease_symptom['label1'][i]['Symptom'] = dict()
    count += 1

count = 0
for i in [ '上呼吸道感染', '小儿支气管炎']:
    classify[i] = 'label2'
    disease_symptom['label2'][i] = dict()
    disease_symptom['label2'][i]['index'] = count
    disease_symptom['label2'][i]['Symptom'] = dict()
    count += 1


goal_train_set = goal_set['dev'] + goal_set['train']

for info in goal_set['dev']:
    disease_tag = info['disease_tag']
    label = classify[disease_tag]
    info['group_id'] = label[5::]
    goal = info['goal']

    for symptom, state in goal['explicit_inform_slots'].items():
        disease_symptom[label][disease_tag]['Symptom'][symptom] = disease_symptom[label][disease_tag]['Symptom'].get(symptom, 0) + 1
        slot_set_dict[label][symptom] = slot_set_dict[label].get(symptom, 0) + 1
    for symptom, state in goal['implicit_inform_slots'].items():
        disease_symptom[label][disease_tag]['Symptom'][symptom] = disease_symptom[label][disease_tag]['Symptom'].get(symptom, 0) + 1
        slot_set_dict[label][symptom] = slot_set_dict[label].get(symptom, 0) + 1
    
    goal_set_HRL[label]['validate'].append(info)
    goal_set_HRL_total['validate'].append(info)

for info in goal_set['train']:
    disease_tag = info['disease_tag']
    label = classify[disease_tag]
    info['group_id'] = label[5::]
    goal = info['goal']

    for symptom, state in goal['explicit_inform_slots'].items():
        disease_symptom[label][disease_tag]['Symptom'][symptom] = disease_symptom[label][disease_tag]['Symptom'].get(symptom, 0) + 1
        slot_set_dict[label][symptom] = slot_set_dict[label].get(symptom, 0) + 1
    for symptom, state in goal['implicit_inform_slots'].items():
        disease_symptom[label][disease_tag]['Symptom'][symptom] = disease_symptom[label][disease_tag]['Symptom'].get(symptom, 0) + 1
        slot_set_dict[label][symptom] = slot_set_dict[label].get(symptom, 0) + 1
    
    goal_set_HRL[label]['train'].append(info)
    goal_set_HRL_total['train'].append(info)

for info in goal_test_set['test']:
    disease_tag = info['disease_tag']
    label = classify[disease_tag]
    info['group_id'] = label[5::]
    goal = info['goal']
    goal_set_HRL[label]['test'].append(info)
    goal_set_HRL_total['test'].append(info)

for i in ['label1','label2']:
    slot_set_dict[i] = dict(sorted(slot_set_dict[i].items(),key=lambda x:x[1],reverse=True))
    slot_set_key[i] = set(slot_set_dict[i].keys())
    for key, value in enumerate(slot_set_key[i]):
        slot_set_dict[i][value] = key




pickle.dump(goal_set_HRL_total , open('.//HRL//goal_set.p', 'wb'))

for i in ['label1','label2' ]:
    pickle.dump(goal_set_HRL[i] , open('.//HRL//'  + i +  '//goal_set.p', 'wb'))
    pickle.dump(slot_set_dict[i] , open('.//HRL//'  + i +  '//slot_set.p', 'wb'))
    pickle.dump(disease_symptom[i] , open('.//HRL//'  + i +  '//disease_symptom.p', 'wb'))