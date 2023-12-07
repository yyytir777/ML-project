import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
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

# MBTI 레이블을 네 가지 차원으로 분리
df['E_I'] = df['label'].apply(lambda x: 1 if x[0].upper() in ['E'] else 0)
df['S_N'] = df['label'].apply(lambda x: 1 if x[1].upper() in ['S'] else 0)
df['T_F'] = df['label'].apply(lambda x: 1 if x[2].upper() in ['T'] else 0)
df['P_J'] = df['label'].apply(lambda x: 1 if x[3].upper() in ['P'] else 0)

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

# 예측 결과를 저장할 딕셔너리 생성
predictions = {}
accuracy_results = {}

# 각 MBTI 차원별로 모델 훈련 및 예측
for dimension in ['E_I', 'S_N', 'T_F', 'P_J']:
    # XGBoost 모델 초기화 및 훈련
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=7
    )
    model.fit(np.array(X_train), y_train[dimension])

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(np.array(X_test))

    # 예측 결과 저장
    predictions[dimension] = y_pred

    # 정확도 계산 및 저장
    acc = accuracy_score(y_test[dimension], y_pred)
    accuracy_results[dimension] = acc

# 예측 결과를 .pkl 파일로 저장
with open('xg_predictions.pkl', 'wb') as file:
    pickle.dump(predictions, file)

# 각 차원별 정확도 출력
for dimension, acc in accuracy_results.items():
    print(f"{dimension} 정확도: {acc:.4f}")

print("예측 결과가 'xg_predictions.pkl' 파일에 저장되었습니다.")

