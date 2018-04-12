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
def derivSigmoid(x):
    return sigmoid(x) * ( 1-sigmoid(x) )
def nn(input,labels,actiFunc, deriv, layer2num,layer3num,learnRate,reguPara): #nn for binary classification
    numPara=layer2num*inputDim+layer3num*layer2num+layer3num+layer2num+ layer3num +1
    inputDim=input.shape[1]
    num=input.shape[0]
    vAct = np.vectorize(actiFunc)
    vDerivAct=np.vectorize(deriv)
    weight=[]
    weight.append(np.random.normal(0,0.01,(layer2num,inputDim)))
    weight.append(np.random.normal(0, 0.01, (layer3num, layer2num)))
    weight.append(np.random.normal(0, 0.01, (1, layer3num)))
    bias=[]
    bias.append(np.random.normal(0,0.01,(layer2num,1)))
    bias.append(np.random.normal(0, 0.01, (layer3num, 1)))
    bias.append(np.random.normal(0, 0.01, (1, 1)))
    while True:
        aAll=[];zAll=[] #forward prop
        for i in range(0,num):
          a=[];z=[]
          a.append(np.reshape(input[i,:],(2,1)))
          z.append(np.reshape(input[i,:],(2,1)))
          for j in range(1,4):
            z.append(np.matmul(weight[j-1],a[j-1])+bias[j-1])
            a.append(vAct(z[j]))
          aAll.append(a)
          zAll.append(z)
            
        dWeight=[]
        dWeight.append(np.zeros((layer2num,inputDim)))
        dWeight.append(np.zeros((layer3num, layer2num)))
        dWeight.append(np.zeros((1,layer3num)))
        dBias.append(np.zeros( (layer2num,1) ))
        dBias.append(np.zeros( (layer3num, 1) ))
        dBias.append(np.zeros( (1, 1) ))
        for k in range(0,num): #backward prop #begin with zero in code,but with 1 in math
          delta=list(range(0,4))
          y=float(labels[k,:])
          zLast=float(zAll[k][3])
          delta[3]=np.array([[ ( y/actiFunc(zLast) + (y-1)/actiFunc(zLast) ) * deriv( zLast ) ]] )
          delta[2]=np.multiply( np.matmul( weight[k][2].transpose(), delta[3] ), vDerivAct( zAll[k][2] ) )
          delta[1]=np.multiply( np.matmul( weight[k][1].transpose(), delta[2] ), vDerivAct( zAll[k][1] ) )
          for l in range(0,3):
            dWeight[l]=dWeight[l]+np.matmul( delta[l+1], aAll[k][l].transpose() )
            dBias[l]=dBias[l]+delta[l+1]
        change=0
        for l in range(0,3):
          changeWeight=learnRate * ( dWeight[l] / num + reguPara * weight[l] )
          change+= np.linalg.norm(changeWeight)**2
          weight[l]=weight[l] - changeWeight 
          changeBias=learnRate * ( dBias[l] / num )
          change+= np.linalg.norm(changeBias)**2
          bias[l] = bias[l] - changeBias
        if change / numPara <0.01:
          return (weight,bias)
        
          
          
        



