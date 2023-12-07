from cosineKNN import KNN
from tf_idf_vectorizer import tf_idf_vectorizer
import pickle
from sklearn.metrics import accuracy_score, classification_report

with open("input_data.pkl", "rb") as read_file:
    input_data = pickle.load(read_file)

with open("target_class.pkl", "rb") as read_file:
    target_class = pickle.load(read_file)


# input data를 tf_idf vector화한다.
tf_idf = tf_idf_vectorizer(input_data)


# 5배수를 기준으로 train data와 test data로 나눈다.
train_x = []
train_y = []
test_x = []
test_y = []


for i in range(len(input_data)):
    if i % 5 == 0 :
        test_x.append(tf_idf[i])
        test_y.append(target_class[i])
    else :
        train_x.append(tf_idf[i])
        train_y.append(target_class[i])


# test data는 전부 사용하고
# train data는 knn의 특성을 고려해서 같은 개수만큼만 사용하고자 한다.
# 가장 적은 것이 63개 있으므로 모든 것에서 63개씩 뽑아서 만든다.

# target_class 에 포함된 mbti별 index 저장
mbti_index = {}

# index를 저장한다.
index = 0

for mbti_name in train_y:
    if mbti_name not in mbti_index.keys():
        mbti_index[mbti_name] = [] # 해당하는 MBTI를 처음 만나면 mbti_index 배열을 초기화한다.

    mbti_index[mbti_name].append(index)
    index += 1

train_x_index = []

for i in mbti_index:
    for j in range(63):
        train_x_index.append(mbti_index[i][j])

# train_x, train_y, test_x, test_y값을 갱신한다.
train_x = []
train_y = []

for i in train_x_index :
    train_x.append(tf_idf[i])
    train_y.append(target_class[i])




# k 값과 weight 값 설정
k = 3
weight = True
num_of_class = 2 # e또는 i의 2개 class 존재


# e와 i를 분류하기 위한 것

e_i_train_y = []
e_i_test_y = []

for label in train_y:
    if label[0] == 'e' :
        e_i_train_y.append(1)
    else:
        e_i_train_y.append(0)

for label in test_y:
    if label[0] == 'e' :
        e_i_test_y.append(1)
    else :
        e_i_test_y.append(0)

knn_e_i = KNN(k, train_x, e_i_train_y, test_x, e_i_test_y, num_of_class, weight)
y_pred = knn_e_i.run()

# 정확도 평가
accuracy = accuracy_score(e_i_test_y, y_pred)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
report = classification_report(e_i_test_y, y_pred, target_names=['i', 'e'])
print("분류 보고서:\n", report)

print(y_pred[0:50])
print(e_i_test_y[0:50])





# s와 n를 분류하기 위한 것
s_n_train_y = []
s_n_test_y = []

for label in train_y:
    if label[1] == 's' :
        s_n_train_y.append(1)
    else:
        s_n_train_y.append(0)

for label in test_y:
    if label[1] == 's' :
        s_n_test_y.append(1)
    else :
        s_n_test_y.append(0)

knn_s_n = KNN(k, train_x, s_n_train_y, test_x, s_n_test_y, num_of_class, weight)
y_pred = knn_s_n.run()

# 정확도 평가
accuracy = accuracy_score(s_n_test_y, y_pred)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
report = classification_report(s_n_test_y, y_pred, target_names=['n', 's'])
print("분류 보고서:\n", report)

print(y_pred[0:50])
print(s_n_test_y[0:50])



# t와 f를 분류하기 위한 것
t_f_train_y = []
t_f_test_y = []

for label in train_y:
    if label[2] == 't' :
        t_f_train_y.append(1)
    else:
        t_f_train_y.append(0)

for label in test_y:
    if label[2] == 't' :
        t_f_test_y.append(1)
    else :
        t_f_test_y.append(0)

knn_t_f = KNN(k, train_x, t_f_train_y, test_x, t_f_test_y, num_of_class, weight)
y_pred = knn_t_f.run()

# 정확도 평가
accuracy = accuracy_score(t_f_test_y, y_pred)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
report = classification_report(t_f_test_y, y_pred, target_names=['f', 't'])
print("분류 보고서:\n", report)

print(y_pred[0:50])
print(t_f_test_y[0:50])





# p와 j를 분류하기 위한 것
p_j_train_y = []
p_j_test_y = []

for label in train_y:
    if label[3] == 'j' :
        p_j_train_y.append(1)
    else:
        p_j_train_y.append(0)

for label in test_y:
    if label[3] == 'j' :
        p_j_test_y.append(1)
    else :
        p_j_test_y.append(0)

knn_p_j = KNN(k, train_x, p_j_train_y, test_x, p_j_test_y, num_of_class, weight)
y_pred = knn_p_j.run()

# 정확도 평가
accuracy = accuracy_score(p_j_test_y, y_pred)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
report = classification_report(p_j_test_y, y_pred, target_names=['p', 'j'])
print("분류 보고서:\n", report)

print(y_pred[0:50])
print(p_j_test_y[0:50])
