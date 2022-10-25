# Trajectory Predict

###TrajectoryPredict_ver1
- window = 5, stride = 1 
- 300개 궤적에 대해 LSTM 학습을 반복


### TrajectoryPredict_ver2
- window = 50, stride = 50 -> 50개의 데이터를 갖는 궤적 하나에 대해 하나의 Y를 갖도록 학습
- 중첩해서 할 필요 없이, 다른 위치의 궤적을 예측할 수 있음