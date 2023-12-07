import numpy as np


def tf_idf_vectorizer(input_data):
    # input 데이터와 output 데이터 pkl 파일을 읽어서 불러들인다.
    # TF-IDF 배열 만들기 1단계 Document-Term Matrix을 만든다.
    # 각 단계에서 Bag_of_words를 만든 것들을 합쳐서 만든다.
    # s번째 인덱스보다 작은 단어들만 사용하겠다.
    # 이 s값은 정확도에 따라 변화시키겠다.
    s = 30000

    document_term_matrix = np.zeros((7811, s), dtype = int)

    index = 0

    for data in input_data :
        for word in data :
            if word < s :
                document_term_matrix[index][word] += 1

        index += 1


    # TF-IDF 배열 만들기 2단계 DF를 만들겠다.
    # 특정 단어 t가 등장한 문서의 수를 구한다.

    df = np.zeros(s, dtype = int)

    for data in document_term_matrix:
        for i in range(s):
            if data[i] != 0 :
                df[i] += 1


    # idf값을 구한다.
    # idf값은 numpy의 행렬 연산을 이용해서 빠르게 수행한다.
    # 총 문서 수 total_doc_num
    total_doc_num = 7811

    idf = np.log(total_doc_num / (1 + df))

    # tf와 idf를 곱해서 tf_idf 행렬을 구한다.
    tf = document_term_matrix

    tf_idf = tf * idf

    return tf_idf