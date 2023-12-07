import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# 데이터 불러오기
df = pd.read_csv("twitter_MBTI.csv", encoding='iso-8859-1')

# 데이터 전처리
data_list = []
n = len(df)  # 데이터셋 전체 사용

for i in range(n):
    string = df.iloc[i, 1]
    splited_string = string.split('|||')
    cleaned_sentences = []
    for sentence in splited_string:
        sentence = sentence.replace('(', '').replace(')', '').replace('?', '').replace('@', '').replace('http', '')
        words = sentence.split()
        words = [word.lower() for word in words if len(word) > 2 and word.isascii()]
        cleaned_sentences.extend(words)
    data_list.append(cleaned_sentences)

# 정수 인코딩
word_count = {}
for words in data_list:
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

sorted_word_list = sorted(word_count, key=word_count.get, reverse=True)
low_freq_words = {word for word, count in word_count.items() if count <= 2}
pro_dic = {word: i + 1 for i, word in enumerate(sorted_word_list) if word not in low_freq_words}
pro_dic['OOV'] = len(pro_dic) + 1

user_list = [[pro_dic.get(word, pro_dic['OOV']) for word in sentence] for sentence in data_list]

# 크기 정규화
user_padded_list = [sentence[:1000] if len(sentence) >= 1000 else sentence + [0] * (1000 - len(sentence)) for sentence in user_list]

# MBTI 레이블을 네 가지 차원으로 정확히 분리하는 함수 정의
def extract_dimension(label, index):
    return 1 if label[index].upper() in ['E', 'S', 'T', 'P'] else 0

# MBTI 레이블을 네 가지 차원으로 분리
df['E_I'] = df['label'].apply(lambda x: extract_dimension(x, 0))
df['S_N'] = df['label'].apply(lambda x: extract_dimension(x, 1))
df['T_F'] = df['label'].apply(lambda x: extract_dimension(x, 2))
df['P_J'] = df['label'].apply(lambda x: extract_dimension(x, 3))

# 데이터셋의 인덱스 생성
indices = np.arange(len(df))

# 5의 배수 인덱스를 테스트 데이터로, 나머지를 훈련 데이터로 분할
test_indices = indices[indices % 5 == 0]
train_indices = indices[indices % 5 != 0]

# 훈련 데이터와 테스트 데이터 분할
X_train = [user_padded_list[i] for i in train_indices]
X_test = [user_padded_list[i] for i in test_indices]
y_train = df[['E_I', 'S_N', 'T_F', 'P_J']].iloc[train_indices]
y_test = df[['E_I', 'S_N', 'T_F', 'P_J']].iloc[test_indices]

# 예측 결과 및 성능 평가를 저장할 딕셔너리 생성
predictions = {}
performance = {}

# 각 MBTI 차원별로 모델 훈련 및 예측
for dimension in ['E_I', 'S_N', 'T_F', 'P_J']:
    # 랜덤 포레스트 모델 초기화 및 훈련
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train[dimension])

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test)

    # 예측 결과 및 성능 평가 저장
    predictions[dimension] = y_pred
    performance[dimension] = {
        'accuracy': accuracy_score(y_test[dimension], y_pred),
        'precision': precision_score(y_test[dimension], y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test[dimension], y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test[dimension], y_pred, average='macro', zero_division=0)
    }

# 예측 결과를 .pkl 파일로 저장
with open('rf_predictions.pkl', 'wb') as file:
    pickle.dump(predictions, file)

# 성능 평가 결과 출력
for dimension, scores in performance.items():
    print(f"{dimension}: {scores}")

print("예측 결과와 성능 평가가 'rf_predictions.pkl' 파일에 저장되었습니다.")

