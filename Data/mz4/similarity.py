import json
import pickle
import numpy as np
from matplotlib import pyplot as plt

slot_set = pickle.load(file=open('./100symptoms/slot_set.p', "rb"))
disease_symptom = pickle.load(file=open('./100symptoms/disease_symptom.p', "rb"))

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = cos
    return sim

def vector_generate(disease_symptom, slot_set):
    vector = []
    for key, value in slot_set.items():
        vector.append(disease_symptom['Symptom'].get(key, 0))
    return vector

similarity_matrix = np.zeros((4, 4))
slot_vector = []
disease_list = ['上呼吸道感染','小儿支气管炎','小儿消化不良','小儿腹泻']
for dis in disease_list:
    slot_vector.append(vector_generate(disease_symptom[dis], slot_set))

for i in range(len(slot_vector)):
    for j in range(len(slot_vector)):
        similarity_matrix[i, j] = cos_sim(slot_vector[i], slot_vector[j])

fig, ax0 = plt.subplots(1, 1)

c = ax0.pcolor(similarity_matrix)
ax0.set_title('Similarity matrix')
fig.colorbar(c, ax=ax0)

plt.savefig('.//100symptoms/similiarity.png', format='png')
pass