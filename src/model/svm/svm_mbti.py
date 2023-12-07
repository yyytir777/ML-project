#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터 로드
twitter_data_path = 'twitter_MBTI.csv'  
encoded_data_path = 'encode_data.pkl'  

twitter_data = pd.read_csv(twitter_data_path)

with open(encoded_data_path, 'rb') as file:
    encoded_data = pickle.load(file)

# MBTI 레이블을 네 가지 차원으로 분리하는 함수
def extract_dimension(label, index):
    return 1 if label[index].upper() in ['E', 'S', 'T', 'P'] else 0

# MBTI 레이블을 네 가지 차원으로 분리
twitter_data['E_I'] = twitter_data['label'].apply(lambda x: extract_dimension(x, 0))
twitter_data['S_N'] = twitter_data['label'].apply(lambda x: extract_dimension(x, 1))
twitter_data['T_F'] = twitter_data['label'].apply(lambda x: extract_dimension(x, 2))
twitter_data['P_J'] = twitter_data['label'].apply(lambda x: extract_dimension(x, 3))

# 데이터 준비
X = encoded_data
y = twitter_data[['E_I', 'S_N', 'T_F', 'P_J']]

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 각 차원별로 SVM 모델 훈련 및 평가
results_svm = {}
dimensions = ['E_I', 'S_N', 'T_F', 'P_J']

# 차원별 예측 결과를 저장할 리스트
predictions = {'E_I': [], 'S_N': [], 'T_F': [], 'P_J': []}


for dim in dimensions:
    # 데이터 분할
    y_dim = twitter_data[dim]
    indices = np.arange(len(X_scaled))
    test_indices = indices[indices % 5 == 0]  # 5의 배수 인덱스
    train_indices = indices[indices % 5 != 0]  # 나머지 인덱스

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y_dim.iloc[train_indices], y_dim.iloc[test_indices]

    # SVM 모델 훈련
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # 예측
    y_pred = svm_model.predict(X_test)
    
    predictions[dim] = y_pred

    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 결과 저장
    results_svm[dim] = {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1_score': f1
    }

# 최종 MBTI 유형을 나타내는 리스트 생성
final_predictions = []
for i in range(len(predictions['E_I'])):
    mbti_type = [
        predictions['E_I'][i],
        predictions['S_N'][i],
        predictions['T_F'][i],
        predictions['P_J'][i]
    ]
    final_predictions.append(mbti_type)

# 결과를 svm_predict.pkl 파일로 저장
with open('svm_predict.pkl', 'wb') as file:
    pickle.dump(final_predictions, file)

# 결과 출력
for dim, result in results_svm.items():
    print(f"Results for {dim}:")
    print(result)
    print()


# In[5]:



# svm_predict.pkl 파일에서 예측 결과 불러오기
with open('svm_predict.pkl', 'rb') as file:
    predictions = pickle.load(file)

# 예측 결과의 차원 크기 확인
print("예측 결과의 차원 크기:", len(predictions))


# In[7]:


from sklearn.metrics import classification_report

# svm_predict.pkl 파일에서 예측 결과 불러오기
with open('svm_predict.pkl', 'rb') as file:
    final_predictions = pickle.load(file)

# 테스트 데이터셋 추출 (5의 배수 인덱스를 사용)
test_indices = np.arange(len(X_scaled)) % 5 == 0
test_labels = y.iloc[test_indices]

# 예측 결과와 실제 레이블을 DataFrame으로 변환
predictions_df = pd.DataFrame(final_predictions, columns=['E_I', 'S_N', 'T_F', 'P_J'])
labels_df = test_labels.reset_index(drop=True)

# 각 차원별로 예측 결과 평가
for dim in dimensions:
    print(f"성능 평가 - {dim}:")
    print(classification_report(labels_df[dim], predictions_df[dim]))

# 통합된 MBTI 유형 평가
def combine_dimensions(row):
    return ''.join([str(int(row['E_I'])), str(int(row['S_N'])), str(int(row['T_F'])), str(int(row['P_J']))])

combined_predictions = predictions_df.apply(combine_dimensions, axis=1)
combined_labels = labels_df.apply(combine_dimensions, axis=1)

print("통합된 MBTI 유형에 대한 성능 평가:")
print(classification_report(combined_labels, combined_predictions))


# In[ ]:




