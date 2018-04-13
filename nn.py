#S1:2  S3:1
# not linearly seperable
# best hyperparameter for 10 neurons classify correctly
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
    return 1 - 1/(1 + np.exp(gamma))
  else:
    return 1/(1 + np.exp(-gamma))
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
    return np.tanh(x)
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
    vLast=np.vectorize(sigmoid,otypes=[float])   # actiFunc and deriv only apply to last layer
    vDerivLast=np.vectorize(derivSigmoid,otypes=[float])
    vAct = np.vectorize(actiFunc,otypes=[float])  # only use relu for  acti other than last layer
    vDerivAct = np.vectorize(deriv,otypes=[float])
    weight=[]
    weight.append(np.random.normal( 0,0.01,(layer2num,inputDim)) )
    weight.append(np.random.normal(0,0.01,(1, layer2num)) )
    bias=[]
    bias.append(np.random.normal(0,0.01,(layer2num,1)) )
    bias.append(np.random.normal(0,0.01,(1, 1)))

    while True:
        batch = random.sample(range(0, num), batchSize)
        aAll=[];zAll=[] #forward prop
        for i in range(0,num):
          a=[];z=[]
          z.append(np.reshape(input[i, :], (2, 1)))
          a.append(np.reshape(input[i,:],(2,1)))
          z.append(np.matmul(weight[0],a[0])+bias[0])
          a.append(vAct(z[1]))
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

        for k in range(0,num): #backward prop #begin with zero in code,but with 1 in math
          delta=list(range(0,3))
          y=float(labels[k,:])
          zLast=float(zAll[k][2])
          delta[2]=np.array([[ -( y/sigmoid(zLast) + (y-1)/(1-sigmoid(zLast)) ) * derivSigmoid( zLast ) ]] )
          #delta[2]=np.array([[ 2*(sigmoid(zLast)-y) * derivSigmoid( zLast ) ]] )
          delta[1]=np.multiply( np.matmul( weight[1].transpose(), delta[2] ), vDerivAct( zAll[k][1] ) )
          for l in range(0,2):
            dWeight[l]=dWeight[l]+np.matmul( delta[l+1], aAll[k][l].transpose() )
            dBias[l]=dBias[l]+delta[l+1]
        change=0
        for l in range(0,2):
          changeWeight=learnRate * ( (dWeight[l] / num) + (reguPara * weight[l]) )
          change+= np.linalg.norm(changeWeight)**2
          weight[l]=weight[l] - changeWeight 
          changeBias=learnRate * ( dBias[l] / num )
          change+= np.linalg.norm(changeBias)**2
          bias[l] = bias[l] - changeBias
          pass

        if (change / numPara) <stopChange or count>=350:
          return (weight,bias,count) #nn(data['X_train'],data['Y_train'],derivSigmoid, sigmoid, 2,1,0.1,0.1)
        # if
        #   return (weight, bias,count)
        count+=1

def predict(model, input, actiFunc):
    num = input.shape[0]
    vLast = np.vectorize(sigmoid,otypes=[float])
    vAct=np.vectorize(actiFunc,otypes=[float])
    weight=model[0]
    bias=model[1]
    aAll = []
    zAll = []  # forward prop
    for i in range(0, num):
        a = []
        z = []
        z.append(np.reshape(input[i, :], (2, 1)))
        a.append(np.reshape(input[i, :], (2, 1)))
        z.append(np.matmul(weight[0], a[0]) + bias[0])
        a.append(vAct(z[1]))
        z.append(np.matmul(weight[1], a[1]) + bias[1])
        a.append(vLast(z[2]))
        aAll.append(a)
        zAll.append(z)
        pass
    labels=[]
    for k in range(0,num):
        labels.append(aAll[k][2])
    return labels


def toLabels(predicts):
  result = []
  for i in predicts:
    p=float(i)
    if p>0.5:
        result.append(1)
    else:
        result.append(0)
  return result

def getAcc(l1,l2):
    count=0
    for i in range(0,len(l1)):
        if l1[i]==l2[i]:
            count+=1
    return count/len(l1)
# trainNn(input,labels,actiFunc, deriv, layer2num,learnRate,reguPara,stopChange, batchSize)

# print("results:")
# print( getAcc( toLabels(predict(model,scaledTst,relu)), data['Y_test'].flatten().tolist() ) )
# print( getAcc(toLabels(predict(model,scaledTrn,relu)), data['Y_train'].flatten().tolist() ) )
# print(model[0])
# print(model[1])
# print(model[2])
#'X_validation', 'X_train', 'X_test', 'Y_train', 'Y_validation', 'Y_test'

import matplotlib.pyplot as plt
trnX1=data['X_train'][:,0]
trnX2=data['X_train'][:,1]
trnClass=data["Y_train"].flatten().tolist()
for i in range(0,len(trnClass)): #0 corresponds to color red,1:blue
    if trnClass[i]==0:
        trnClass[i]="red"
    else:
        trnClass[i] = "blue"

# trnClass2=data["Y_train"].flatten().tolist()
# for i in range(0,len(trnClass2)): #0 corresponds to color green,1:red
#     if trnClass2[i]==0:
#         trnClass2[i]="s"
#     else:
#         trnClass2[i] = "o"
plt.scatter(trnX2,trnX1,c=trnClass)
plt.show()
func=relu  # specify actiFunc here
actFunc=derivRelu # specify actiFunc( derivative )  here
def cv(nueron):
  reguDict={}
  for reguPara in (0, 0.0001, 0.01, 0.1, 1, 10):
    model = trainNn(scaledTrn, data['Y_train'], func, actFunc, nueron, 1, reguPara, 0.000000001, 200)
    reguDict[reguPara]=getAcc(toLabels(predict(model, scaledValid, func)), data['Y_validation'].flatten().tolist())
  print(str(nueron)+" neuron CV result: ")
  print(reguDict)
  for i in reguDict:
    max=0
    if reguDict[i]>=max:
      winner=i
  model = trainNn(scaledTrn, data['Y_train'], func, actFunc, nueron, 1, winner, 0.000000001, 200)

  print("winner parameter: "+str(winner)+" training error: ")
  print(getAcc(toLabels(predict(model, scaledTrn, func)), data['Y_train'].flatten().tolist()))
  print("winner parameter test error:")
  print(getAcc(toLabels(predict(model, scaledTst, func)), data['Y_test'].flatten().tolist()))

cv(2)
cv(10)


