import pickle
import pandas

### 피클 파일 불러오기 ###
with open("./data/222.116.156.173-001-20221003-060000-010121-010131_2.mp4.pickle","rb") as fr:
    data = pickle.load(fr)

