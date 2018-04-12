import numpy as np
from sklearn import preprocessing
import math
import random
import scipy.io
mat = scipy.io.loadmat('/Users/yzh/Downloads/dataset.mat')
print('balanced class in training? class 1 ratio:'+str(mat['Y_train'].flatten().mean()))
data={}
for i in ('X_validation', 'X_train', 'X_test', 'Y_train', 'Y_validation', 'Y_test'): #transpose format
    data[i]=mat[i].transpose()

scaler = preprocessing.StandardScaler().fit(data["X_train"]) #feature scaling
scaledTrn   = scaler.transform(data["X_train"])
scaledTst=scaler.transform(data["X_test"])
scaledValid=scaler.transform(data["X_validation"])
def sigmoid(x):
    return 1/(1+math.exp(-x))

def nn(input,label,actiFunc,layer2num,layer3num): #nn for binary classification
    inputDim=input.shape[1]
    vAct = np.vectorize(actiFunc)
    weight=[]
    weight.append(np.random.normal(0,0.01,(layer2num,inputDim)))
    weight.append(np.random.normal(0, 0.01, (layer3num, layer2num)))
    weight.append(np.random.normal(0, 0.01, (1, layer3num)))
    bias=[]
    bias.append(np.random.normal(0,0.01,(layer2num,1)))
    bias.append(np.random.normal(0, 0.01, (layer3num, 1)))
    bias.append(np.random.normal(0, 0.01, (1, 1)))
    while True:
        aAll=[];zAll=[]
        for i in range(0,input.shape[0]):
          a=[];z=[]
          a.append(np.reshape(input[i,:],(2,1)))
          z.append(np.reshape(input[i,:],(2,1)))
          for j in range(1,4):
            z.append(np.matmul(weight[j-1],a[j-1])+bias[j-1])
            a.append(vAct(z[j]))
          aAll.append(a)
          zAll.append(z)



