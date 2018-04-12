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
def sigmoidPre(gamma):# avoid overflow
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))
def sigmoid(x):  #constrain to 0 to 1 to avoid zeroDevision
    pre=sigmoidPre(x)
    if pre==0:
        return 10**(-16)
    elif pre==1:
        return 1-10**(-16)
    else:
        return pre
def derivSigmoid(x):
    return sigmoid(x) * ( 1-sigmoid(x) )
def tanh(x):
    return (math.exp(2*x)-1)/(math.exp(2*x)+1)
def derivTanh(x):
    return 1-tanh(x)**2
def relu(x):
    if x>0:
        return x
    else:
        return 0
def derivRelu(x):
    if x>0:
        return 1
    else:
        return 0
def trainNn(input,labels,actiFunc, deriv, layer2num,learnRate,reguPara,stopChange, batchSize): #nn for binary classification
    count=0
    inputDim=input.shape[1] #gradient vanishing?
    numPara = layer2num * inputDim  + 2*layer2num  + 1
    num=input.shape[0]
    vLast=np.vectorize(actiFunc)   # actiFunc and deriv only apply to last layer
    vDerivLast=np.vectorize(deriv)
    vAct = np.vectorize(relu)  # only use relu for  acti other than last layer
    vDerivAct = np.vectorize(derivRelu)
    weight=[]
    weight.append(np.random.normal( 0,1,(layer2num,inputDim)) )
    weight.append(np.random.normal(0,1,(1, layer2num)) )
    bias=[]
    bias.append(np.random.normal(0,1,(layer2num,1)) )
    bias.append(np.random.normal(0,1,(1, 1)))

    while True:
        batch = random.sample(range(0, num), batchSize)
        aAll=[];zAll=[] #forward prop
        for i in batch:
          a=[];z=[]
          a.append(np.reshape(input[i,:],(2,1)))
          z.append(np.reshape(input[i,:],(2,1)))
          for j in range(1,2):
            z.append(np.matmul(weight[j-1],a[j-1])+bias[j-1])
            a.append(vAct(z[j]))
          z.append(np.matmul(weight[1], a[1]) + bias[1])
          a.append(vLast(z[2]))
          aAll.append(a)
          zAll.append(z)
            
        dWeight=[]
        dWeight.append(np.zeros((layer2num,inputDim)))
        dWeight.append(np.zeros((1,layer2num)))
        dBias=[]
        dBias.append(np.zeros( (layer2num,1) ))
        dBias.append(np.zeros( (1, 1) ))
        for k in range(0,batchSize): #backward prop #begin with zero in code,but with 1 in math
          delta=list(range(0,3))
          y=float(labels[batch[k],:])
          zLast=float(zAll[k][2])
          delta[2]=np.array([[ ( y/actiFunc(zLast) + (y-1)/(1-actiFunc(zLast)) ) * deriv( zLast ) ]] )
          delta[1]=np.multiply( np.matmul( weight[1].transpose(), delta[2] ), vDerivAct( zAll[k][1] ) )
          for l in range(0,2):
            dWeight[l]=dWeight[l]+np.matmul( delta[l+1], aAll[k][l].transpose() )
            dBias[l]=dBias[l]+delta[l+1]
        change=0
        for l in range(0,2):
          changeWeight=learnRate * ( dWeight[l] / num + reguPara * weight[l] )
          change+= np.linalg.norm(changeWeight)**2
          weight[l]=weight[l] - changeWeight 
          changeBias=learnRate * ( dBias[l] / num )
          change+= np.linalg.norm(changeBias)**2
          bias[l] = bias[l] - changeBias
          pass

        # if (change / numPara)**0.5 <stopChange:
        #   return (weight,bias) #nn(data['X_train'],data['Y_train'],derivSigmoid, sigmoid, 2,1,0.1,0.1)
        if count>3:
          return (weight, bias)
        count+=1

def predict(model, input, actiFunc):
    num = input.shape[0]
    vLast = np.vectorize(actiFunc)
    vAct=np.vectorize(relu)
    weight=model[0]
    bias=model[1]
    aAll = []
    zAll = []  # forward prop
    for i in range(0, num):
        a = []
        z = []
        a.append(np.reshape(input[i, :], (2, 1)))
        z.append(np.reshape(input[i, :], (2, 1)))
        for j in range(1, 2):
            z.append(np.matmul(weight[j - 1], a[j - 1]) + bias[j - 1])
            a.append(vAct(z[j]))
        z.append(np.matmul(weight[1], a[1]) + bias[1])
        a.append(vLast(z[2]))
        aAll.append(a)
        zAll.append(z)
        pass
    labels=[]
    for k in range(0,num):
        labels.append(aAll[k][2])
    return labels
# trainNn(input,labels,actiFunc, deriv, layer2num,learnRate,reguPara,stopChange, batchSize)
model=trainNn(scaledTrn,data['Y_train'],sigmoid,derivSigmoid,10,0.1,0,0.0001,200)
print(predict(model,scaledTrn,sigmoid))


result=[]
for i in predict(model, scaledTrn, sigmoid):
    p=float(i)
    if p>0.5:
        result.append(1)
    else:
        result.append(0)
print(np.array(result)==data['Y_train'].flatten())
print(model[0])
print(model[1])