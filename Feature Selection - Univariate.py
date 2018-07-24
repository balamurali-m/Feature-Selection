# -*- coding: utf-8 -*-
"""
Univariate Feature Selection
Author: Balamurali M
"""

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import warnings
warnings.filterwarnings('ignore')

#Generating matrix with random explanatory and response variables
matr = np.random.randint(200, size=(100, 20))
print (matr.shape)

train_exp = matr[:80, :19]
train_res = matr[:80, 19:]
test_exp = matr[80:, :19]
test_act = matr[80:, 19:]

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
     
    def FS_UNI(self):
        a1 = SelectKBest(chi2, k=4)
        fit1 = a1.fit_transform(self.w1, self.x1)
        return fit1
        
        
matr_exp = FS_ABC(train_exp, train_res, test_exp, test_act)
fit2 = matr_exp.FS_UNI()
print (fit2) #highest scoring features