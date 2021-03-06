import cupy as cp
import numpy as np
import argparse
import scipy.linalg
from opencifar100 import load_cifar
import math
np.set_printoptions(threshold=10000)

samples = 1
sample_type = 100
train_sample_type = 0
train_samples = 0

parser = argparse.ArgumentParser(description = 'Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10')
parser.add_argument('--depth', default = 21, type = int, help = 'depth of CNTK (#conv layers + 1)')
parser.add_argument('--gap', default = "yes", type = str, help = 'whether GAP (global average pooling) is used')
parser.add_argument('--fix', default = "yes", type = str, help = 'whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)')
args = parser.parse_args()

d = args.depth
gap = (args.gap == "yes")
fix = (args.fix == "yes")

def normalize_list(list):
    max_value = max(list)
    min_value = min(list)
    for i in range(0, len(list)):
        list[i] = (list[i] - min_value + 0.001) / (max_value - min_value)
    return list

#CUDA kernel for convolution operation
conv3 = cp.RawKernel(r'''
extern "C" __global__
void conv3(const float s[84][84][84][84], float t[84][84][84][84])
{
	int x1 = threadIdx.x + blockIdx.x - 83;
	int y1 = threadIdx.y + blockIdx.y - 83;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	__shared__ float d[84 + 2][84 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[85][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][85] = d[85][0] = d[85][85] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][85] = 0;
	}
	if (x1 < 0 || x1 > 83 || y1 < 0 || y1 > 83){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();
	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];
}''', 'conv3')

conv3check = cp.RawKernel(r'''
extern "C" __global__
void conv3check(const float s[84][84][84][84], float t[84][84][84][84], float D[84 + 2][84 + 2]) 
{
	int x1 = threadIdx.x + blockIdx.x - 83;
	int y1 = threadIdx.y + blockIdx.y - 83;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	__shared__ float d[84 + 2][84 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[85][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][85] = d[85][0] = d[85][85] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][85] = 0;
	}
	if (x1 < 0 || x1 > 83 || y1 < 0 || y1 > 83){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	  D[x2 + 2][y2 + 2] = s[x1][y1][x2][y2];
	__syncthreads();
	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];
}''', 'conv3check')

conv_blocks = (63, 63)
conv_threads = (32, 32)

#CUDA kernel for activation
trans = cp.RawKernel(r'''
extern "C" __global__
void trans(float s[84][84][84][84], float t[84][84][84][84], const float l[84][84], const float r[84][84], const float il[84][84], const float ir[84][84])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
	S = S * iL * iR;
	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T * S + BS;
	s[x1][y1][x2][y2] = BS;
}''', 'trans')
trans_blocks = (32, 32, 16)
trans_threads = (8, 8)

#Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper. 
def xx(x):
	RL = [1.0, ]
	iRL = [1.0, ]

	S = cp.matmul(x.T, x).reshape(84, 84, 84, 84)
	D = cp.zeros((84, 84), dtype = cp.float32)
	conv3check(conv_blocks, conv_threads, (S, S, D))
	T = cp.zeros((84, 84, 84, 84), dtype = cp.float32)
	if not fix:
		T += S

	for i in range(1, d - 1):
		#cupy.diag only take diagonal array output length 1024
		#cupy.sqrt Elementwise square root
		L = cp.sqrt(cp.diag(S.reshape(7056, 7056)).reshape(84, 84))
		iL = 1.0 / L
		RL.append(L)
		iRL.append(iL)
		trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))

	L = cp.sqrt(cp.diag(S.reshape(7056, 7056)).reshape(84, 84))
	TL = cp.sqrt(cp.average(S.reshape(7056, 7056) , axis= 0).reshape(84, 84))
	# tempeate change , don't mind
	L = TL
	iL = 1.0 / L
	RL.append(L)
	iRL.append(iL)
	trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))	
	
	if fix:
		T -= S
	return RL, iRL, TL

#Caclulate the kernel value of x and z.
#Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
#iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
def xz(x, z, Lx, Lz, iLx, iLz, Y1, Y2, TLsi, TLsj):
	tmp = []
	IB = []
	
	S = cp.matmul(x.T, z).reshape(84, 84, 84, 84)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((84, 84, 84, 84), dtype = cp.float32)
	if not fix:
		T += S
	xy = []
	xx = []
	yy = []

	for i in range(1, d - 1):
		trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))
		#print("layer",i, "x y",cp.mean(T),"xx yy",cp.mean(Lx[i]), cp.mean(Lz[i]),
		#      "result",np.log(1-(cp.mean(Lx[i]) * cp.mean(Lz[i]) / cp.mean(T) * cp.mean(T))),
		#      (1-(cp.mean(Lx[i]) * cp.mean(Lz[i]) / cp.mean(T) * cp.mean(T)))
		xy.append(cp.mean(T))
		xx.append(cp.mean(Lx[i]))
		yy.append(cp.mean(Lz[i]))
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))
		tmp.append(S)
		
		#print("layer",i , ":",(1-(cp.mean(Lx[i]) * cp.mean(Lz[i]) / cp.mean(T) * cp.mean(T))))

	trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))
	tmp.append(S)
	xy.append(cp.mean(S))
	xx.append(cp.mean(Lx[i]))
	yy.append(cp.mean(Lz[i]))
	xy1 = normalize_list(xy)
	xx1 = normalize_list(xx)
	yy1 = normalize_list(yy)
	if Y1==Y2:
		res = [(1-(cp.mean(xx1[i] * yy1[i] / xy1[i] * xy1[i]))) for i in range(len(xx1))]
		index = res.index(max(res))
		print(res, index)
	else:
		res = [(1-(cp.mean(xx1[i] * yy1[i] / xy1[i] * xy1[i]))) for i in range(len(xx1))]
		index = res.index(min(res))
		res = [1-item for item in res]
		print(res, index)
	if fix:
		T -= S
	#cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))
	#cp.mean(cp.linalg.eigh(T.reshape(1024, 1024))[0])
	total = 0
	for i,element in enumerate(res):
		total = total + element * tmp[i]
	return cp.mean(total) if gap else cp.trace(total.reshape(7056, 7056))


from random import sample
import random

#Load CIFAR-10.
(X_train, y_train), (X_test, y_test) = load_cifar()

deadlist = []
sample_type = [i for i in range(sample_type)]
print(sample_type)
random.shuffle(sample_type)
print(sample_type)

for it in sample_type:
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
print(sample_type)
random.shuffle(sample_type)
print(sample_type)
for it in sample_type:
	x = 0
	tmp = []
	for index,item in enumerate(y_test):
		if item==it:
			x = x + 1
			tmp.append(index)
		if x >= (samples*100):
			#tmp = tmp[50:55]
			tmp = sample(tmp, 1)
			break
	deadlist = deadlist + tmp		


X_test  = X_test[deadlist,:,:,:]
y_test = y_test[deadlist]

print("X_train",X_train.shape,"X_test",X_test.shape)
X = np.concatenate((X_train, X_test), axis = 0)
Y = np.concatenate((y_train, y_test), axis = 0)
N = X.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X = cp.asarray(X).reshape(-1, 3, 1024)
print(X.shape)

#Calculate diagonal entries.
L = []
iL = []
TLs = [] 
for i in range(N):
	Lx, iLx,TL = xx(X[i])	
	L.append(Lx)
	iL.append(iLx)
	TLs.append(TL)

#####Calculate kernel values.
#####Below we provide a naive implementation using for-loops.
#####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.
H = np.zeros((N, N), dtype = np.float32)
for i in range(N):
	for j in range(N):
		H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j],Y[i], Y[j],TLs[i],TLs[j])
#####

print(H)
#Solve kernel regression.
Y_train = np.ones((N_train, 100)) * -0.1
for i in range(N_train):
	Y_train[i][y_train[i]] = 0.9
u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
print(np.argmax(u, axis = 1))
print(y_test)
print("test accuracy:", 1.0 * np.sum(np.argmax(u, axis = 1) == y_test) / N_test)

import pickle
f = open("samples" + str(samples) +"layer" + str(d) +'.txt', 'wb')
list_row = H
pickle.dump(list_row, f)
