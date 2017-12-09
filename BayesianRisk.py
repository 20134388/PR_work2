#-- coding:utf-8 --
import numpy as np
from generateData import *
from LikelihoodRate import *

K=3

def getBayesLabel(PosterPro):
    Cost = [[0, 2, 3], [1, 0, 5], [1, 1, 0]]
    M=np.shape(PosterPro)[0]
    BayesLabel = np.zeros(M)
    for m in range(M):
        for i in range(K):
            flag = True
            for j in range(i + 1, K):
                temp = (Cost[j][i] - Cost[i][i]) * np.array(PosterPro)[m][i] - \
                       (Cost[i][j] - Cost[j][j]) * np.array(PosterPro)[m][j]
                if temp < 0: flag = False
            if flag == True:
                BayesLabel[m] = i
                break
            else:
                BayesLabel[m] = j
                continue
    return BayesLabel


'''
errorNum=0
for i in range(len(label)):
    if label[i]!=outputLabel[i]:
        errorNum+=1
errorRate[iter]=float(errorNum) / N
'''

#data1:
PosterPro_data1=getPosterPro(K,data1,sigma_X1,mean_X1,PrioPro_data1)
BayesLabel_data1=getBayesLabel(PosterPro_data1)
errorRate_X1_Bayesian=getErrorRate(N,label_data1,BayesLabel_data1)

#data2
PosterPro_data2=getPosterPro(K,data2,sigma_X2,mean_X2,PrioPro_data2)
BayesLabel_data2=getBayesLabel(PosterPro_data2)
errorRate_X2_Bayesian=getErrorRate(N,label_data2,BayesLabel_data2)
