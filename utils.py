import pickle
import numpy as np
import os

#sample number for evey type
samples = 200


def load_cifar(path = "cifar-10-batches-py"):
	train_batches = []
	train_labels = []

	for i in range(1, 6):
		strs = os.path.join(path, "data_batch_{0}".format(i))
		print(strs)
		with open(strs, 'rb') as f:
			cifar_out = pickle.load(f,encoding='latin1')
	#	cifar_out = pickle.load(open(strs, encoding='latin1'))
		print(cifar_out.keys())
		train_batches.append(cifar_out["data"])
		train_labels.extend(cifar_out["labels"])
	X_train_total= np.vstack(tuple(train_batches)).reshape(-1, 3, 32, 32)
	y_train_total = np.array(train_labels)
	X_train= []
	y_train= []
	i1 = 0
	i2 = 0

	for index,item in enumerate(y_train_total):
		if item == 0:
			if i1 >= samples:
				continue
			X_train.append(X_train_total[index,:,:,:])
			y_train.append(y_train_total[index])
			i1 = i1 + 1
		if item == 1:
			if i2 >= samples:
				continue
			X_train.append(X_train_total[index,:,:,:])
			y_train.append(y_train_total[index])
			i2 = i2 + 1

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	print(X_train.shape)
	print(y_train.shape)


	with open(os.path.join(path, "test_batch"), 'rb') as f:
		cifar_out = pickle.load(f,encoding='latin1')
	#cifar_out = pickle.load(open(os.path.join(path, "test_batch")))
	X_test_total = cifar_out["data"].reshape(-1, 3, 32, 32)
	y_test_total = cifar_out["labels"]

	X_test= []
	y_test= []
	i1 = 0
	i2 = 0
	for index,item in enumerate(y_test_total):
		if item == 0:
			if i1 >= samples:
				continue
			X_test.append(X_test_total[index,:,:,:])
			y_test.append(y_test_total[index])
			i1 = i1 + 1
		if item == 1:
			if i2 >= samples:
				continue
			X_test.append(X_test_total[index,:,:,:])
			y_test.append(y_test_total[index])
			i2 = i2 + 1
	X_test = np.asarray(X_test)
	y_test = np.asarray(y_test)
	print(X_test.shape)
	print(y_test.shape)
 
	mean = X_train.mean(axis = (0, 2, 3)) 
	std = X_train.std(axis = (0, 2, 3)) 
	X_train = (X_train - mean[:, None, None]) / std[:, None, None]
	X_test = (X_test - mean[:, None, None]) / std[:, None, None]

	return (X_train, np.array(y_train)), (X_test, np.array(y_test))
