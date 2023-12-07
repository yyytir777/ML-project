import numpy as np

# KNN class
class KNN():
  def __init__(self, k, x_train, y_train, x_test, y_test, num_of_class, isWeighted = False):
    self.k = k
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.num_of_class = num_of_class
    self.isWeighted = isWeighted

  # cosine 유사도를 구하는 함수이다.
  def cosine_similarity(self, a, b):
    sum_of_a_mul_b = np.sum(a * b)
    sum_of_a_square = np.sum(a * a)
    sum_of_b_square = np.sum(b * b)
    similarity = sum_of_a_mul_b / (np.sqrt(sum_of_a_square) * np.sqrt(sum_of_b_square))
    return similarity

  # 두 데이터 i와 j에 대해 코사인 거리를 계산하는 함수이다.
  # 코사인 거리 = 1 - 코사인 유사도
  def cosine_distance(self, i, j):
    distance = 1 - self.cosine_similarity(i, j)
    return distance

  # 모든 데이터에 대한 코사인 거리를 구해서 넣어 놓는다.
  def calc_distance(self):
    dist_array = np.zeros((np.shape(self.x_test)[0], np.shape(self.x_train)[0]))
    for i in range(np.shape(self.x_test)[0]) :
      for j in range(np.shape(self.x_train)[0]) :
        dist_array[i][j] = self.cosine_distance(self.x_test[i], self.x_train[j])

    return dist_array

  # knn을 구하는 함수 obtain_knn
  def obtain_knn(self, dist_array):
    knn_index_array = np.zeros((np.shape(self.x_test)[0], self.k), dtype= int)
    knn_dist_array = np.zeros((np.shape(self.x_test)[0], self.k), dtype = int)
    for i in range(np.shape(self.x_test)[0]) :
      knn_index = np.argsort(dist_array[i]) # dist_array의 값이 낮은 순부터 index를 적는다.
      knn_index = knn_index[0:self.k] # 값이 낮은 k개만 knn_index로 정한다.
      knn_index_array[i] = knn_index # test 데이터 i에 대한 knn의 index를 정한다.
      knn_dist_array[i] = dist_array[i][knn_index] #dist_array에서 knn_index를 갖는 것들만 모아서 knn_dist_array로 삼는다.

    return knn_index_array, knn_dist_array

  # majority vote
  # 하나의 test 데이터와 그것의 knn에 해당하는 train data에서의 인덱스를 받아서 이 test 데이터의 레이블 값을 구해본다.
  # 레이블은 0, 1, 2, 3, 4, ...의 숫자값으로 들어와야한다.
  def majority_vote(self, knn_index_arr):
    count = np.zeros(self.num_of_class) # class 숫자만큼 카운트 배열을 만든다.

    # knn_y 는 knn index에 해당하는 y의 레이블들의 배열이다
    knn_y = []
    for i in knn_index_arr:
      knn_y.append(self.y_train[i])

    # 배열의 각각의 원소에 대해서 카운트 값을 1 증가시킨다.
    for i in knn_y:
      count[i] += 1

    # majority vote를 통해 얻어진 k값 중 가장 큰 값을 리턴한다.
    result = np.argmax(count)
    return result

  # weighted majority vote
  # 하나의 테스트 데이터와 그것의 knn에 해당하는 train data에서의 인덱스와 그 인덱스에서의 값을 받아서 이 test 데이터의 레이블 값을 구해본다.
  # weighted majority vote를 사용한다.
  def weighted_majority_vote(self, knn_index_arr, knn_dist_arr):
    score = np.zeros(self.num_of_class) # score는 거리와의 역수를 리턴한다.

    # knn_y 는 knn index에 해당하는 y의 레이블들의 배열이다
    knn_y = []
    for i in knn_index_arr:
      knn_y.append(self.y_train[i])

    for i in range(len(knn_y)):
      knn_dist_arr = knn_dist_arr + 1e-7
      score[knn_y[i]] += 1 / knn_dist_arr[i]

    result = np.argmax(score)
    return result

  # 실제로 모든 함수를 작동시켜서 결과를 내는 run 함수
  def run(self):
    dist_arr = self.calc_distance() # 모든 test 데이터와 train 데이터 사이의 거리를 계산한다.
    knn_index_arr, knn_dist_arr = self.obtain_knn(dist_arr) # 모든 테스트 데이터 각각에 대한 knn을 얻는다.

    prediction = [] # test_y에 대한 예측값들을 넣어놓은 것들

    for i in range(len(self.x_test)):
      if self.isWeighted :
        result = self.weighted_majority_vote(knn_index_arr[i], knn_dist_arr[i])
      else :
        result = self.majority_vote(knn_index_arr[i])

      prediction.append(result)

    return prediction