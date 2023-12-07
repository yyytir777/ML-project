import pickle
import numpy as np

with open("knn_predict_data.pkl", "rb") as knn_f:
    knn = pickle.load(knn_f)

with open("svm_predict_data.pkl", "rb") as svm_f:
    svm = pickle.load(svm_f)

with open("rnn_predict_data.pkl", "rb") as rnn_f:
    rnn = pickle.load(rnn_f)

with open("xg_predict_data.pkl", "rb") as xg_f:
    temp_xg = pickle.load(xg_f)

with open("rf_predict_data.pkl", "rb") as rf_f:
    temp_rf = pickle.load(rf_f)


# rnn_predict_data.pkl 데이터 정렬
rnn = rnn.astype(np.int32)


index_list = ['E_I', 'S_N', 'T_F', 'P_J']


# xg_predict_data.pkl 데이터 정렬
xg = list()
cnt = 0
for i in range(len(temp_xg['E_I'])):
    temp = list()
    for j in index_list:
        temp.append(temp_xg[j][cnt])
    xg.append(temp)
    cnt += 1


# rf_predict_data.pkl 데이터 정렬
rf = list()
cnt = 0
for i in range(len(temp_rf['E_I'])):
    temp = list()
    for j in index_list:
        temp.append(temp_rf[j][cnt])
    rf.append(temp)
    cnt += 1

knn = np.array(knn)
svm = np.array(svm)
rnn = np.array(rnn)
xg = np.array(xg)
rf = np.array(rf)

# 모델 predict 값 shape 출력
# print(knn.shape)
# print(svm.shape)
# print(rnn.shape)
# print(xg.shape)
# print(rf.shape)

# 모델 predict 값 출력
# print("knn : \n", knn[:100])
# print("svm : \n", svm[:100])
# print("rnn : \n", rnn[:100])
# print("xg : \n", xg[:100])
# print("rf : \n", rf[:100])

def majority_vote(lst):
    return max(lst, key=lst.count)


# 앙상블 진행
model_list = [knn, rnn, xg, rf]
ensemble_predict_list = list()
for i in range(len(knn)):
    first_feature = majority_vote(list([knn[i][0], svm[i][0], rnn[i][0], xg[i][0], rf[i][0]]))
    second_feature = majority_vote(list([knn[i][1], svm[i][1], rnn[i][1], xg[i][1], rf[i][1]]))
    third_feature = majority_vote(list([knn[i][2], svm[i][2], rnn[i][2], xg[i][2], rf[i][2]]))
    forth_feature = majority_vote(list([knn[i][3], svm[i][3], rnn[i][3], xg[i][3], rf[i][3]]))

    ensemble_predict_list.append([first_feature, second_feature, third_feature, forth_feature])


with open("y_test.pkl", "rb") as fr:
    y_test = pickle.load(fr)

ensemble_predict_list = np.array(ensemble_predict_list)
y_test = np.array(y_test)

cnt = 0
temp_sum = 0
dimension_cnt = [0, 0, 0, 0]
test_data_size = 1563
for i in range(test_data_size):
    temp = 0
    for j in range(4):
        if ensemble_predict_list[i][j] == y_test[i][j]:
            dimension_cnt[j] += 1
            temp += 1
    
    temp_sum += temp
    if temp == 4:
        cnt += 1

print("cnt : %d" %cnt)
print("size : %d" %test_data_size)
print("accuracy : %0.3f" %(cnt / test_data_size))
print()
print("temp : %d" %temp_sum)
print("size : %d" %(test_data_size * 4))
print("accuracy : %0.3f" %(temp_sum / (test_data_size * 4)))
print()
for i in range(4):
    print("%d 차원 정확도 : %0.3f" %(i, (dimension_cnt[i] / test_data_size)))