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

# (e, i), (s, n)를 구할 때는 majority vote에 k = 5 사용
# k는 5, 클래스의 개수는 2개인 KNN을 사용한다.
k = 5
weight = False
num_of_class = 2 # e또는 i의 2개 class 존재


# e와 i를 구분하기 위해서 e를 0, i를 1로 레이블링 한다.
e_i_train_y = []
e_i_test_y = []

for label in train_y:
    if label[0] == 'e' :
        e_i_train_y.append(0)
    else:
        e_i_train_y.append(1)

for label in test_y:
    if label[0] == 'e' :
        e_i_test_y.append(0)
    else :
        e_i_test_y.append(1)

# e, i에 대한 KNN을 통한 y의 예측값을 구한다.
knn_e_i = KNN(k, train_x, e_i_train_y, test_x, e_i_test_y, num_of_class, weight)
y_pred_ei = knn_e_i.run()


# s와 n를 분류하기 위해 s를 0, n을 1로 레이블링 한다.
s_n_train_y = []
s_n_test_y = []

for label in train_y:
    if label[1] == 's' :
        s_n_train_y.append(0)
    else:
        s_n_train_y.append(1)

for label in test_y:
    if label[1] == 's' :
        s_n_test_y.append(0)
    else :
        s_n_test_y.append(1)

# s와 n에 대한 KNN을 통한 y의 예측값을 구한다.
knn_s_n = KNN(k, train_x, s_n_train_y, test_x, s_n_test_y, num_of_class, weight)
y_pred_sn = knn_s_n.run()


# (t, f), (p, j)를 구할 때는 weighted majority vote에 k = 3 사용
k = 3
weight = True

# t와 f를 분류하기 위해 t를 0, f을 1로 레이블링 한다.
t_f_train_y = []
t_f_test_y = []

for label in train_y:
    if label[2] == 't' :
        t_f_train_y.append(0)
    else:
        t_f_train_y.append(1)

for label in test_y:
    if label[2] == 't' :
        t_f_test_y.append(0)
    else :
        t_f_test_y.append(1)

# t와 f에 대한 KNN을 통한 y의 예측값을 구한다.
knn_t_f = KNN(k, train_x, t_f_train_y, test_x, t_f_test_y, num_of_class, weight)
y_pred_tf = knn_t_f.run()


# p와 j를 분류하기 위해 p를 0, j를 1로 레이블링 한다.
p_j_train_y = []
p_j_test_y = []

for label in train_y:
    if label[3] == 'p' :
        p_j_train_y.append(0)
    else :
        p_j_train_y.append(1)

for label in test_y:
    if label[3] == 'p':
        p_j_test_y.append(0)
    else:
        p_j_test_y.append(1)

# p와 j에 대한 KNN을 통한 y의 예측값을 구한다.
knn_p_j = KNN(k, train_x, p_j_train_y, test_x, p_j_test_y, num_of_class, weight)
y_pred_pj = knn_p_j.run()

y_pred = []

for i in range(len(y_pred_ei)):
    y_pred.append([y_pred_ei[i], y_pred_sn[i], y_pred_tf[i], y_pred_pj[i]])

with open('knn_predict_data_new.pkl', 'wb') as file:
    pickle.dump(y_pred, file)

print(y_pred_ei[0:10])
print(y_pred_sn[0:10])
print(y_pred_tf[0:10])
print(y_pred_pj[0:10])
print(y_pred[0:10])

new_test_y = []
new_pred_y = []

for i in range(len(test_y)):
    new_test_y.append(8 * e_i_test_y[i] + 4 * s_n_test_y[i] + 2 * t_f_test_y[i] + 1 * p_j_test_y[i])
    new_pred_y.append(8 * y_pred[i][0] + 4 * y_pred[i][1] + 2 * y_pred[i][2] + 1 * y_pred[i][3])

# 정확도 평가
accuracy = accuracy_score(e_i_test_y, y_pred_ei)
print("e - i")
print(f'정확도: {accuracy:.2f}')

# 정확도 평가
accuracy = accuracy_score(s_n_test_y, y_pred_sn)
print("s - n")
print(f'정확도: {accuracy:.2f}')

# 정확도 평가
accuracy = accuracy_score(t_f_test_y, y_pred_tf)
print("t - f")
print(f'정확도: {accuracy:.2f}')

# 정확도 평가
accuracy = accuracy_score(p_j_test_y, y_pred_pj)
print("p - j")
print(f'정확도: {accuracy:.2f}')


# 정확도 평가
accuracy = accuracy_score(new_test_y, new_pred_y)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
report = classification_report(new_test_y, new_pred_y, target_names=[
    'estp', 'estj', 'esfp', 'esfj', 'entp', 'entj', 'enfp', 'enfj',
    'istp', 'istj', 'isfp', 'isfj', 'intp', 'intj', 'infp', 'infj'])
print("분류 보고서:\n", report)