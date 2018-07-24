# -*- coding: utf-8 -*-
"""
Variance Threshold
Author: Balamurali M
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')

#Generating matrix
matr = np.random.randint(2, size=(10, 4))
print (matr.shape)

train_exp = matr[:8, :3]
train_res = matr[:8, 3:]
test_exp = matr[8:, :3]
test_act = matr[8:, 3:]

print('train_exp',train_exp.shape)
print('train_res',train_res.shape)
print('test_exp',test_exp.shape)
print('test_act',test_act.shape)

class FS_ABC:
    def __init__(self, w1, x1, y1, z1):
        self.w1 = w1
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1 
     
    def FS_VT(self):
        a1 = VarianceThreshold(threshold=(.21)) #0.7 X 0.3
        fit1 = a1.fit_transform(self.w1)
        return fit1
    
matr_exp = FS_ABC(train_exp, train_res, test_exp, test_act)
fit2 = matr_exp.FS_VT()
print (fit2)