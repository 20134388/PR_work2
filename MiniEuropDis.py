#--coding:utf-8--
from numpy import *
from math import *
from generateData import *
from LikelihoodRate import getErrorRate,errorRate_data1_likelihood,errorRate_data2_likelihood
from BayesianRisk import errorRate_X1_Bayesian,errorRate_X2_Bayesian
from MaxPostePro import errorRate_data1_MaxPost,errorRate_data2_MaxPost
N=1000
K=3
def getDistanced(vec1,vec2):
    return sqrt(sum(power(vec1-vec2,2)))

def getLabel(data,K,mean_X):
    m,n=shape(data)
    clusterAssment=zeros(m)
    centroids=mat(mean_X)
    for i in range(m):
        minDist=inf
        minIndex=-1
        for j in range(K):
            dist=getDistanced(data[i,:],centroids[j,:])
            if dist<minDist:
                minDist=dist
                minIndex=j
        if clusterAssment[i]!=minIndex:
            clusterAssment[i]=minIndex
    return centroids,clusterAssment

centroids1,clusterAssment1=getLabel(data1,K,mean_X1)
errorRate_X1=getErrorRate(N,label_data1,clusterAssment1)

centroids2, clusterAssment2 = getLabel(data2, K,mean_X2)
errorRate_X2 = getErrorRate(N, label_data2, clusterAssment2)


print 'The error rate of Likelihood Rate rule is(X1)',errorRate_data1_likelihood
print 'The error rate of Bayesian risk rule is (X1)',errorRate_X1_Bayesian
print 'The error rate of Maximum Posteriori Probability rule is(X1)',errorRate_data1_MaxPost
print 'The error rate of Minimum European distance rule(X1) is ',errorRate_X1
print '\n'
print 'The error rate of Likelihood Rate rule is(X2)',errorRate_data2_likelihood
print 'The error rate of Bayesian risk rule is (X2)',errorRate_X2_Bayesian
print 'The error rate of Maximum Posteriori Probability rule is(X2)',errorRate_data2_MaxPost
print 'The error rate of Minimum European distance rule(X2) is ',errorRate_X2