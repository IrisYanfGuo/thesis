import toolkit as tk
import pandas as pd
import numpy as np
from AI_knn import *
from naive_bayes_conti import *
import seaborn as sns
import matplotlib.pyplot as plt
from naive_bayes import *
from AI_id3 import *
from time import time


def cross_vali_split_data(dataset, cv=10):
    '''

    :param dataset: type matrix
    :param cv: the number of validation fold
    :return: K-1 length list contain the dataset
    '''
    # 向下取整
    size = int(np.floor(len(dataset) / cv))
    result = []
    begin = 0
    end = size
    for i in range(cv - 1):
        testset = dataset[begin:end, :]
        trainset = np.row_stack((dataset[0:begin, :], dataset[end:, :]))
        begin = end
        end = end + size
        result.append([trainset, testset])
    result.append([dataset[:begin, :], dataset[begin:, :]])
    return result


def accuracy_score(right, predict):
    num = 0
    for i in range(len(right)):
        if right[i] == predict[i]:
            num += 1
    return num / len(right)


def mcc(predict, right, classi):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] == classi:
            if right[i] == classi:
                TP += 1
            else:
                FP += 1
        else:
            if right[i] == classi:
                FN += 1
            else:
                if predict[i] == right[i]:
                    TN += 1
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 0.0001)

    dict = {}
    dict['MCC'] = MCC
    dict['TP'] = TP
    dict['FP'] = FP
    return dict


def MCC_TP_FP_dict(predict, right):
    key_list = []
    for i in right:
        if i in key_list:
            pass
        else:
            key_list.append(i)

    MCC_TP_FP = {}
    for i in key_list:
        MCC_TP_FP[i] = mcc(predict, right, i)
    return MCC_TP_FP


def kappa(mat):
    total = sum(sum(mat))

    po = sum([mat[i][i] for i in range(len(mat))]) / total

    pe = 0
    for i in range(len(mat)):
        pec = sum(mat[:, i]) * sum(mat[i, :]) / total ** 2
        pe += pec

    kappa_value = (po - pe) / (1 - pe)
    return kappa_value


def classify_map(predict, right):
    key_list = []
    for i in right:
        if i in key_list:
            pass
        else:
            key_list.append(i)

    map_dict = {}
    for i in key_list:
        dict_t = {}
        for j in key_list:
            dict_t[j] = 0
        map_dict[i] = dict_t
    for i in range(len(right)):
        if (predict[i] != ""):
            map_dict[right[i]][predict[i]] += 1

    map = pd.DataFrame(map_dict)
    return map


def cross(model, attr, dataset, cvfold=10):
    np.random.shuffle(dataset)
    right = dataset[:, -1]
    train_test_pair_list = cross_vali_split_data(dataset, cvfold)
    prediction = []
    train_time = 0
    pre_time = 0
    if model == id3:

        for train_test in train_test_pair_list:
            train_start = time()
            mod = model(list(attr), (train_test[0]).tolist())
            train_end = time()
            train_time += train_end - train_start
            for i in mod.getPrediction(train_test[1].tolist())[1]:
                prediction.append(i)
            pre_end = time()
            pre_time += pre_end - train_end
    else:

        for train_test in train_test_pair_list:
            train_start = time()
            mod = model(attr, train_test[0])
            train_end = time()
            train_time += train_end - train_start
            for i in mod.getPrediction(train_test[1])[1]:
                prediction.append(i)
            pre_end = time()
            pre_time += pre_end - train_end

    accuracy = accuracy_score(right, prediction)
    MCC_TP_FP = MCC_TP_FP_dict(prediction, right)
    classi_map = classify_map(prediction, right)
    kappa_score = kappa(classi_map.as_matrix())
    return accuracy, MCC_TP_FP, kappa_score, classi_map, train_time, pre_time


def draw_map(map, xlab="predict result", ylab="right "):
    sns.heatmap(map, annot=True)
    sns.plt.xlabel(xlab)
    sns.plt.ylabel(ylab)
    plt.show()


def nice_print_model_info(accuracy, dict, kappa_score):
    print("Kappa statistic: ", kappa_score)
    print("Overall Accuracy: ", accuracy)
    for i in dict.keys():
        print(i, "  TP", dict[i]['TP'], "  FP", dict[i]['FP'], "  MCC", dict[i]['MCC'])



'''

car_attr, car = tk.readDataSet("./dataset/iris.csv")

#car_attr_num, car_num = tk.read_catgorical("./dataset/blood.csv")
# acc, mcca, ka, map,t1,t2 = cross(Nai, car_attr, car)
# nice_print_model_info(acc,mcca,ka)
# draw_map(map)




# a comparison of Naive Bayes and Knn and id3
# test stability
accu_list_naive = []
ka_list_naive = []
train_time_naive = []
pre_time_naive = []
mcc_list_naive = []

for i in range(10):
    acc, mcca, ka, map, t1, t2 = cross(Naive_bayes_conti, car_attr, car, 5)
    accu_list_naive.append(acc)
    ka_list_naive.append(ka)
    train_time_naive.append(t1)
    pre_time_naive.append(t2)
    mcc_list_naive.append(mcca)
'''

'''
accu_list = []
ka_list = []
train_time = []
pre_time = []
mcc_list = []

for i in range(10):
    acc, mcca, ka, map2, t1, t2 = cross(knn, car_attr, car, 5)
    accu_list.append(acc)
    ka_list.append(ka)
    mcc_list.append(mcca)
    train_time.append(t1)
    pre_time.append(t2)

'''


'''
df = pd.DataFrame(
    { "naive bayes train time": train_time_naive,
     "naive bayes predict time": pre_time_naive, "KNN train time": train_time, "KNN predict time": pre_time})
df.plot()
plt.xlabel("test index")
plt.ylabel("time")
plt.title("Time cost of naive bayes and KNN")
plt.show()


df_accu = pd.DataFrame(
    {"naive bayes accuracy": accu_list_naive,
     "naive bayes kappa": ka_list_naive,
     "knn accuracy": accu_list, "knn kappa:": ka_list})
'''
'''
df_accu.plot.hist(alpha=0.7, bins=5)
plt.xlabel("accuracy")
plt.title("Accuracy of knn and naive bayes")
plt.show()
'''

'''

df_accu.plot(ylim=[0, 1])
plt.xlabel("test index")
plt.ylabel("Evaluation of algorithms on Iris dataset")
plt.show()


'''