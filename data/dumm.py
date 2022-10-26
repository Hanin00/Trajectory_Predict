import pickle
import pandas as pd


with open('./dumm.pickle', 'rb') as f:
    data = pickle.load(f)

print(data.head())

