'''
数据处理模块
'''

import pickle
import json
import copy
import numpy as np

goal_set = pickle.load(open('./goal_set.p', 'rb'))
goal_test_set = pickle.load(open('./goal_test_set.p', 'rb'))
info = copy.deepcopy(goal_set)
info['test'] = goal_test_set['test']
info['all'] = goal_set['train'] + goal_test_set['test']
#print(info)


'''
生成goal_dict_original_dxy.txt & test_goal_dict_dxy.txt & train_goal_dict_dxy.txt
'''
result_all = {}
train_list = []
test_list = []
all_list = []
temp = {}
for i in info['train']:
    temp['request_slots'] = {'disease':'UNK'}
    temp['implicit_inform_slots'] = i['goal']['implicit_inform_slots']
    temp['explicit_inform_slots'] = i['goal']['explicit_inform_slots']
    temp['disease_tag'] = i['disease_tag']
    temp['consult_id'] = i['consult_id']
    train_list.append(copy.deepcopy(temp))   # 注意：这里要深拷贝，不然最后结果全都一样


for j in info['test']:
    temp['request_slots'] = {'disease':'UNK'}
    temp['implicit_inform_slots'] = j['goal']['implicit_inform_slots']
    temp['explicit_inform_slots'] = j['goal']['explicit_inform_slots']
    temp['disease_tag'] = j['disease_tag']
    temp['consult_id'] = j['consult_id']
    test_list.append(copy.deepcopy(temp))

all_list = train_list + test_list


result_all['all'] = all_list
result_all['train'] = train_list
result_all['test'] = test_list

# 保存结果部分

file = open('dataset_dxy/goal_dict_original_dxy.txt', 'w')
js = json.dumps(result_all, ensure_ascii=False)  #防止中文乱码
file.write(js)
file.close()

file = open('dataset_dxy/test_goal_dict_dxy.txt', 'w')
js = json.dumps(test_list, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/train_goal_dict_dxy.txt', 'w')
js = json.dumps(train_list, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/goal_dict_original_dxy.p', 'wb')
pkl = pickle.dumps(result_all)
file.write(pkl)
file.close()

file = open('dataset_dxy/test_goal_dict_dxy.p', 'wb')
pkl = pickle.dumps(test_list)
file.write(pkl)
file.close()

file = open('dataset_dxy/train_goal_dict_dxy.p', 'wb')
pkl = pickle.dumps(train_list)
file.write(pkl)
file.close()


temp = {}
x = 0
l = []
for i in range(10):
    temp['a'] = x
    temp['b'] = x + 1
    temp['c'] = x + 2
    temp['d'] = x + 3
    l.append(copy.deepcopy(temp))
    x += 4




'''
对dise_sym_num_dict.p & req_dise_sym_dict.p进行调研   以及生成dise_sym_num_dict_dxy.p & req_dise_sym_dict.p文件
'''
'''
num = 0 # 统计下各个symptom在各个disease中的数量
for i in info['all']:
    if i['disease_tag'] == '小儿消化不良':
        for key, value in i['explicit_inform_slots'].items():
            if key == '稀便' and value == True:
                num += 1
        for key, value in i['implicit_inform_slots'].items():
            if key == '稀便' and value == True:
                num += 1
print(num)
'''
disease_set = pickle.load(open('./disease_set.p', 'rb'))
disease_set = list(disease_set.keys())

result = dict()
result1 = dict()
for disease in disease_set:
    result[disease] = dict()
    result1[disease] = list()
for i in info['train']:
    # disease
    for disease in disease_set:           
        if i['disease_tag'] == disease:
            for key, value in i['goal']['explicit_inform_slots'].items():
                if value == True and key in result[disease]: # 该symptom已经在result中
                    result[disease][key] += 1
                elif value == True and bool(1-(key in result[disease])):
                    result[disease][key] = 1
            for key, value in i['goal']['implicit_inform_slots'].items():
                if value == True and key in result[disease]: # 该symptom已经在result中
                    result[disease][key] += 1
                elif value == True and bool(1-(key in result[disease])):
                    result[disease][key] = 1


for i in info['test']:
    # 过敏性鼻炎
    for disease in disease_set:           
        if i['disease_tag'] == disease:
            for key, value in i['goal']['explicit_inform_slots'].items():
                if value == True and key in result[disease]: # 该symptom已经在result中
                    result[disease][key] += 1
                elif value == True and bool(1-(key in result[disease])):
                    result[disease][key] = 1
            for key, value in i['goal']['implicit_inform_slots'].items():
                if value == True and key in result[disease]: # 该symptom已经在result中
                    result[disease][key] += 1
                elif value == True and bool(1-(key in result[disease])):
                    result[disease][key] = 1
print(result)
#
for disease in disease_set:
    temp_alergic_rhinitis = sorted(result[disease].items(), key = lambda item:item[1], reverse = True)
    alergic_rhinitis = []
    for i in temp_alergic_rhinitis:
        alergic_rhinitis.append(i[0])
    alergic_rhinitis = alergic_rhinitis[0:10]
    result1[disease] = alergic_rhinitis
#print(result1)

file = open('dataset_dxy/dise_sym_num_dict_dxy.txt', 'w')
js = json.dumps(result, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/dise_sym_num_dict_dxy.p', 'wb')
pkl = pickle.dumps(result)
file.write(pkl)
file.close()


file = open('dataset_dxy/req_dise_sym_dict_dxy.txt', 'w')
js = json.dumps(result1, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/req_dise_sym_dict_dxy.p', 'wb')
pkl = pickle.dumps(result1)
file.write(pkl)
file.close()






'''
对dise_sym_pro.txt & sym_dise_pro.txt & sym_prio.txt调研     以及生成dise_sym_pro_dxy.txt & sym_dise_pro_dxy.txt & sym_prio_dxy.txt文件
'''
''''
d1 = np.loadtxt('dataset/sym_prio.txt')
print(type(d1))
print(d1)

d2 = np.loadtxt('dataset/dise_sym_pro.txt')
d3 = np.loadtxt('dataset/sym_dise_pro.txt')
print(d2.shape, '\n', d3.shape)
'''
d4 = pickle.load(open('./slot_set.p', 'rb'))
sym = list(d4.keys())
print(sym)


# # --------------------------------------------------------统计MZ病例中一共出现的症状数--------------------------------------------------
temp = {}
for i in info['all']:
    for key, value in i['goal']['explicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp :
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['goal']['implicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

print(temp)
temp1 = sorted(temp.items(), key = lambda item:item[1], reverse = True)
print(temp1)
temp2 = []
for i in temp1:
    temp2.append(i[0])
print(temp2)  # 得出结论：symptoms.txt文件中的症状并不是按出现次数的多少进行排序的
print(len(temp2))
x = 0

file = open('dataset_dxy/symptoms_dxy.txt', 'w')
for i in temp2[:-1:]:
    file.writelines(i+'\n')
file.writelines(temp2[-1])

file.close()


# # ----------------------------------------------查看sym_prio.txt以及生成sym_prio_dxy.txt------------------------------------------------
temp = {}
for i in info['all']:
    for key, value in i['goal']['explicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['goal']['implicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

print(temp)
print(len(info['all']))
sym_p = []
sym = temp2
for i in sym:
    sym_p.append(temp[i]/len(info['all']))
sym_p = np.array(sym_p)
sym_p = sym_p.reshape((1, len(sym)))
print(sym_p)
#
np.savetxt('dataset_dxy/sym_prio_dxy.txt', sym_p, fmt = "%f", delimiter = ' ')




# -------------------------------对dise_sym_pro.txt和sym_dise_pro.txt进行调研 并且生成dise_sym_pro_dxy.txt和sym_dise_pro_dxy.txt---------------------


dis = disease_set 
sym = temp2

p_dis_sym_pro = np.zeros((len(dis), len(sym)))
p_sym_dis_pro = np.zeros((len(dis), len(sym)))
# print(p_dis_sym_pro)
print(dis)
print(sym)
# print(dis.index('小儿消化不良\n'))

for i in info['all']:
    for key, value in i['goal']['explicit_inform_slots'].items():
        if (value == True or value == '1'):
            p_dis_sym_pro[dis.index(i['disease_tag'])][sym.index(key)] += 1
    for key, value in i['goal']['implicit_inform_slots'].items():
        if (value == True or value == '1'):
            p_dis_sym_pro[dis.index(i['disease_tag'])][sym.index(key)] += 1


p_sym_dis_pro = np.copy(p_dis_sym_pro)
#print(p_sym_dis_pro)

temp = {}
for i in info['all']:
    for key, value in i['goal']['explicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['goal']['implicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

print(temp)

for i in range(len(p_dis_sym_pro[0])):
    p_dis_sym_pro[:, i] = p_dis_sym_pro[:, i] / temp[sym[i]]
p_dis_sym_pro = np.transpose(p_dis_sym_pro)
print(p_dis_sym_pro)
np.savetxt('dataset_dxy/sym_dise_pro_dxy.txt', p_dis_sym_pro, fmt = "%f", delimiter = ' ')
#
temp1 = {}
for i in info['all']:
    if i['disease_tag'] in temp1:
        temp1[i['disease_tag']] += 1
    elif 1-(i['disease_tag'] in temp1):
        temp1[i['disease_tag']] = 1


print(temp1)

for i in range(len(p_sym_dis_pro)):
    p_sym_dis_pro[i, :] = p_sym_dis_pro[i, :] / temp1[dis[i]]

print(p_sym_dis_pro)
np.savetxt('dataset_dxy/dise_sym_pro_dxy.txt', p_sym_dis_pro, fmt = "%f", delimiter = ' ')



'''
调研action_mat.txt以及生成action_mat_dxy.txt文件
'''
'''
mat = np.loadtxt('dataset/action_mat.txt')
print(mat.shape)
file = open('dataset/slot_set.txt', 'r', encoding = 'utf-8')
temp = file.readlines()
file.close()
print(len(temp))
'''
# 思路是生成四个矩阵，最后拼接成一个大矩阵
'''
file = open('./dataset_dxy/diseases_dxy.txt', 'r', encoding = 'utf-8')
dis = file.readlines()
file.close()
file = open('./dataset_dxy/symptoms_dxy.txt', 'r', encoding = 'utf-8')
sym = file.readlines()
file.close()
print(dis)
print(sym)
'''
dis_dis = np.zeros((len(dis), len(dis)))
#print(dis_dis)
sym_dis = np.loadtxt('./dataset_dxy/dise_sym_pro_dxy.txt')
#print(sym_dis)
dis_sym = np.loadtxt('./dataset_dxy/sym_dise_pro_dxy.txt')
#print(dis_sym)

sym_sym = np.zeros((len(sym), len(sym)))
print(sym_sym)
sym_app = []
for i in info['all']:
    for k_e, v_e in i['goal']['explicit_inform_slots'].items():
        if v_e == True :#and (1 - (k_e in sym_app)):
            sym_app.append(k_e)
    for k_i, v_i in i['goal']['implicit_inform_slots'].items():
        if v_i == True :#and (1 - (k_i in sym_app)):
            sym_app.append(k_i)
    for i in sym_app:
        for j in sym_app:
            if i != j:
                sym_sym[sym.index(i)][sym.index(j)] += 1
    sym_app.clear()

print(sym_sym)

temp = {}
for i in info['all']:
    for key, value in i['goal']['explicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['goal']['implicit_inform_slots'].items():
        if (value == True or value == '1') and key in temp:
            temp[key] += 1
        elif (value == True or value == '1') and (1 - (key in temp)):
            temp[key] = 1

print(temp)

for r in range(len(sym_sym)):
    sym_sym[r, :] = sym_sym[r, :] / temp[sym[r]]

print(sym_sym)

# 拼接矩阵
t_upper_sim = np.concatenate((dis_dis, sym_dis), axis = 1)
t_under_sim = np.concatenate((dis_sym, sym_sym), axis = 1)
action_mat_sim = np.concatenate((t_upper_sim, t_under_sim), axis = 0)
print(action_mat_sim)

# 二次拼接
t1 = np.eye(2, dtype = float)
t2 = np.zeros((2, len(dis) + len(sym)))
t3 = np.zeros((len(dis) + len(sym), 2))
t_upper = np.concatenate((t1, t2), axis = 1)
t_under = np.concatenate((t3, action_mat_sim), axis = 1)
action_mat = np.concatenate((t_upper, t_under), axis = 0)
print(action_mat)
np.savetxt('dataset_dxy/action_mat_dxy.txt', action_mat, fmt = "%f", delimiter = ' ')



'''
调研slot_set和symtoms的区别
'''

# file = open('dataset_dxy/symptoms_dxy.txt', 'r')
# sym = file.readlines()
# file.close()



# file = open('dataset_dxy/diseases_dxy.txt', 'r')
# dis = file.readlines()
# file.close()
#
temp = ['UNK']
#
slot = dis + temp + sym

file = open('dataset_dxy/slot_set_dxy.txt', 'w')
for i in slot[:-1:]:
    file.writelines(i+'\n')
file.writelines(slot[-1])
file.close()

file = open('dataset_dxy/diseases_dxy.txt', 'w')
for i in dis[:-1:]:
    file.writelines(i+'\n')
file.writelines(dis[-1])
file.close()




