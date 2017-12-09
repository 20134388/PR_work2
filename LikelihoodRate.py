#-- coding:utf-8 --
import numpy as np
import math
from generateData import *

K=3

PrioPro_data1 = np.array((n1_data1, n2_data1, n3_data1)) / 1000.
PrioPro_data2 = np.array((n1_data2, n2_data2, n3_data2)) / 1000.
def getPosterPro(K,data,sigma,mu,PrioPro):
    m,n=np.shape(data)
    Px_w = np.mat(np.zeros((m, K)))
    for i in range(K):
        coef = (2 * math.pi) ** (-n / 2.) * (np.linalg.det(sigma[i]) ** (-0.5))
        temp=np.multiply((data-mu[i])*np.mat(sigma[i]).I,data-mu[i])
        Xshift = np.sum(temp, axis=1)
        Px_w[:,i]= coef * np.exp(Xshift*-0.5)  #矩阵与常数相乘
    PosterPro=np.mat(np.zeros((m,K)))
    for i in range(K):
        PosterPro[:,i]=PrioPro[i]*Px_w[:,i]
    return PosterPro

def getLikelihoodLabel(PosterPro):
    outputLabel = np.argmax(PosterPro, axis=1)
    outputLabel = map(int, np.array(outputLabel.flatten())[0])
    return outputLabel

def getErrorRate(N,label,outputLabel):
    errorNum = np.int(np.shape(np.nonzero(np.array(outputLabel) - np.array(label)))[1])
    errorRate = float(errorNum) / N
    return errorRate

#data1
PosterPro_data1=getPosterPro(K,data1,sigma_X1,mean_X1,PrioPro_data1)
likelihoodLabel=getLikelihoodLabel(PosterPro_data1)
errorRate_data1_likelihood=getErrorRate(N,label_data1,likelihoodLabel)

PosterPro_data2=getPosterPro(K,data2,sigma_X2,mean_X2,PrioPro_data2)
likelihoodLabel_data2=getLikelihoodLabel(PosterPro_data2)
errorRate_data2_likelihood=getErrorRate(N,label_data2,likelihoodLabel_data2)
