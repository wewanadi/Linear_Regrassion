# Regrassion
###### tags:`Linear Regrassion` `Ridge Regrassion` `plot`
---
## Outcome
- y = 2x + ε , ε = N(0,1)
- 0.75 training set, 0.25 testing set
- training with 3 different process
  - just Regrassion
  - k fold , k = 5 
  - leave one out

### Different Degree
<div align="center">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/b_1.png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(b_5).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(b_10).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(b_14).png width="220">
</div>

### Different Data
<div align="center">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(b_14).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(d_14_60_1).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(d_14_160_1).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(d_14_320_1).png width="220">
</div>

### Ridge Regration
<div align="center">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(e_14_20_1).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(e_14_20_2).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(e_14_20_3).png width="220">
  <img src=https://github.com/wewanadi/Linear_Regrassion/blob/master/picture/ML_HW1(e_14_20_4).png width="220">
</div>

---
## Code

### Set data
``` python
data_size = 20
test_size = int(data_size*0.25)
train_size = int(data_size*0.75)
import numpy as np

x = np.linspace(0,1,data_size)
ε = np.random.normal(0,1,size=(data_size))
y = 2 * x + ε
```

### Split train and test
``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state =3 ,shuffle=True)
```

### Transform x into X
``` python
def power_mat(x,pol):
    pol += 1
    matrix = np.array([])
    for i in range(len(x)):
        tmp_lst = []
        for p in reversed(range(pol)):
            tmp_lst += [x[i] ** p]
        matrix = np.append(matrix, np.array(tmp_lst))
    matrix = matrix.reshape(-1, pol)
    return matrix
 
pol = 5
X_train = power_mat (X_train,pol)
X_test = power_mat (X_test,pol)
```
