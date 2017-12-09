#--coding:utf-8--
from numpy import *
from math import *
from generateData import *
from LikelihoodRate import getErrorRate

N=1000
K=3
def getDistanced(vec1,vec2):
    return sqrt(sum(power(vec1-vec2,2)))
'''
def getRandomCentroids(data,K):
    m,n=shape(data)
    initCentroids=mat(zeros((K,n)))
    for i in range(n):
        minJ=min(data[:,i])
        rangeJ=float(max(data[:,i])-minJ)
        initCentroids[:,i]=minJ+rangeJ*random.rand(K,1)
    return initCentroids
'''
def Kmeans(data,K,mean_X):
    m,n=shape(data)
    clusterAssment=zeros(m)
    flag=True
    #centroids=getRandomCentroids(data,K)
    centroids=mat(mean_X)
    while flag==True:
        flag=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(K):
                dist=getDistanced(data[i,:],centroids[j,:])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if clusterAssment[i]!=minIndex: flag=True
            clusterAssment[i]=minIndex
        #print centroids
        for j in range(K):
            temp=data[nonzero(clusterAssment==j)]
            centroids[j,:]=mean(temp,axis=0)
        centroids=centroids[lexsort(centroids[:,::-1].T)][0]
    return centroids,clusterAssment


def trainTestSplit(X,test_size):
    m,n=shape(X)
    train_index=range(m)
    test_index=[]
    test_num=int(m*test_size)
    for i in range(test_num):
        randomIndex=int(random.uniform(0,len(train_index)))
        test_index.append(randomIndex)
        del(train_index[randomIndex])
    trainSet=X[train_index]
    testSet=X[test_index]
    return trainSet,testSet

def getTrainTestSet(n1_data,n2_data,n3_data,test_size):
    train_part1,test_part1=trainTestSplit(n1_data,test_size)
    train_part2,test_part2=trainTestSplit(n2_data,test_size)
    train_part3, test_part3 = trainTestSplit(n3_data, test_size)
    trainDataSet=vstack((train_part1,train_part2,train_part3))
    testDataSet=vstack((test_part1,test_part2,test_part3))
    testLabelSet=hstack((zeros(shape(test_part1)[0]),ones(shape(test_part2)[0]),2*ones(shape(test_part3)[0])))
    return trainDataSet,testDataSet,testLabelSet

def getTestSetPredictedLabel(centroids,testDataSet):
    testPredictedLabel=zeros(len(testDataSet))
    for i in range(len(testDataSet)):
        mindist=inf
        for j in range(shape(centroids)[0]):
            dist=getDistanced(centroids[j,:],testDataSet[i,:])
            if dist<mindist:
                mindist=dist
                testPredictedLabel[i]=j
    return testPredictedLabel

#用训练数据来训练模型，用测试集合来计算错误率
#for K in range(2,5):
#迭代执行5次
Iter=5
errorSum1=0;errorSum2=0
for iter in range(Iter):
    trainDataSet_X1, testDataSet_X1 ,testLabelSet_X1= getTrainTestSet(X1_part1.T, X1_part2.T, X1_part3.T, 0.2)
    centroids1,clusterAssment1=Kmeans(trainDataSet_X1,K,mean_X1)
    testPredictedLabel_X1=getTestSetPredictedLabel(centroids1,testDataSet_X1)
    errorRate_X1=getErrorRate(shape(testDataSet_X1)[0],testLabelSet_X1,testPredictedLabel_X1)
    #errorRate_X1=getErrorRate(N,label,clusterAssment1)
    errorSum1+=errorRate_X1
    trainDataSet_X2, testDataSet_X2, testLabelSet_X2 = getTrainTestSet(X2_part1.T, X2_part2.T, X2_part3.T, 0.2)
    centroids2, clusterAssment2 = Kmeans(trainDataSet_X2, K,mean_X2)
    testPredictedLabel_X2 = getTestSetPredictedLabel(centroids2, testDataSet_X2)
    errorRate_X2 = getErrorRate(shape(testDataSet_X2)[0], testLabelSet_X2, testPredictedLabel_X2)
    #errorRate_X2 = getErrorRate(N, label, clusterAssment2)
    errorSum2+=errorRate_X2

print 'The Average Error Rate of Minimum European distance rule(X1) is ',errorSum1/Iter
print 'The Average Error Rate of Minimum European distance rule(X2) is ',errorSum2/Iter