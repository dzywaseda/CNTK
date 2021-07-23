
import cupy as cp
import numpy as np
import argparse
import scipy.linalg
from utilpy3 import load_cifar
np.set_printoptions(threshold=10000)



#Load CIFAR-10.
(X_train, y_train), (X_test, y_test) = load_cifar()
deadlist = []

for it in range(sample_type):
	x = 0
	for index,item in enumerate(y_train):
		if item==it:
			x = x + 1
			deadlist.append(index)
		if x >= samples:
			break
			
for it in range(train_sample_type):
	it = it + 2
	x = 0
	for index,item in enumerate(y_train):
		if item==it:
			x = x + 1
			deadlist.append(index)
		if x >= train_samples:
			break
			
		
		

X_train = X_train[deadlist,:,:,:]
y_train = y_train[deadlist]


deadlist = []
for it in range(sample_type):
	x = 0
	for index,item in enumerate(y_test):
		if item==it:
			x = x + 1
			deadlist.append(index)
		if x >= samples:
			break
			
			
X_test  = X_test[deadlist,:,:,:]
y_test = y_test[deadlist]

print("X_train",X_train.shape,"X_test",X_test.shape)
X = np.concatenate((X_train, X_test), axis = 0)
N = X.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X = cp.asarray(X).reshape(-1, 3, 1024)
print(X.shape)

import pickle
f = open("/result/samples20layer24.txt","rb")
list_row = pickle.load(f)
print(list_row.shape)

f = open("./result/samples20layer9.txt","rb")
list_row2 = pickle.load(f)
print(list_row.shape)

H = list_row + list_row2

#Solve kernel regression.
Y_train = np.ones((N_train, 10)) * -0.1
for i in range(N_train):
	Y_train[i][y_train[i]] = 0.9
u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
print("test accuracy:", 1.0 * np.sum(np.argmax(u, axis = 1) == y_test) / N_test)

