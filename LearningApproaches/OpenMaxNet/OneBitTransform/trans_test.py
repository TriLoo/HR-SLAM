import joblib
import numpy as np
import model

trans = joblib.load('trans_lst.joblib')
thresh = joblib.load('local_thsh.joblib')

ele_len = len(trans)

for i in range(ele_len):
    print(trans[i] - thresh[i])


print('--------------------------')

a = np.linspace(1, 10, 10)
b = model._one_dimension_boxfilter(a, 3)
print(b)
