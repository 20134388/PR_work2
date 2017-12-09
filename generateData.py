#-- coding:utf-8 --

import numpy as np
import random as ran
import matplotlib.pyplot as plt


N=1000

mu1=[1,1]
mu2=[4,4]
mu3=[8,1]
sigma=[[2,0],[0,2]]
n1_data1,n2_data1,n3_data1=0,0,0
#数据集X1
for dataNum in range(0,N):
    k=ran.randint(1,3)
    if k==1:
        n1_data1+=1
    elif k==2:
        n2_data1+=1
    else:
        n3_data1+=1
x1,y1=np.random.multivariate_normal(mu1, sigma, n1_data1).T
x2,y2=np.random.multivariate_normal(mu2,sigma,n2_data1).T
x3,y3=np.random.multivariate_normal(mu3,sigma,n3_data1).T
'''
fig=plt.figure()
ax1=fig.add_subplot(121)
ax1.scatter(x1,y1,c='red')
ax1.scatter(x2,y2,c='blue')
ax1.scatter(x3,y3,c='green')
'''
n1_data2,n2_data2,n3_data2=0,0,0
X1_part1=np.vstack((x1,y1))
X1_part2=np.vstack((x2,y2))
X1_part3=np.vstack((x3,y3))
cov1_data1=np.cov(X1_part1)
cov2_data1=np.cov(X1_part2)
cov3_data1=np.cov(X1_part3)
mean1_data1=np.mean(X1_part1,axis=1)
mean2_data1=np.mean(X1_part2,axis=1)
mean3_data1=np.mean(X1_part3,axis=1)
data1=np.hstack((X1_part1,X1_part2,X1_part3))

#数据集X2
for dataNum in range(0,N):
    k=ran.randint(1,10)
    if k<=6:
        n1_data2+=1
    elif k<=9:
        n2_data2+=1
    else:
        n3_data2+=1
x1,y1=np.random.multivariate_normal(mu1,sigma,n1_data2).T
x2,y2=np.random.multivariate_normal(mu2,sigma,n2_data2).T
x3,y3=np.random.multivariate_normal(mu3,sigma,n3_data2).T

'''
ax2=fig.add_subplot(122)
ax2.scatter(x1,y1,c='red')
ax2.scatter(x2,y2,c='blue')
ax2.scatter(x3,y3,c='green')
plt.show()
'''

X2_part1=np.vstack((x1,y1))
X2_part2=np.vstack((x2,y2))
X2_part3=np.vstack((x3,y3))
cov1_data2=np.cov(X2_part1)
cov2_data2=np.cov(X2_part2)
cov3_data2=np.cov(X2_part3)
mean1_data2=np.mean(X2_part1,axis=1)
mean2_data2=np.mean(X2_part2,axis=1)
mean3_data2=np.mean(X2_part3,axis=1)
data2=np.hstack((X2_part1,X2_part2,X2_part3))

data1=data1.T
data2=data2.T


label_data1 = np.zeros(N)
label_data1[n1_data1:n1_data1 + n2_data1 - 1] = 1
label_data1[N - n3_data1:] = 2
label_data1 = map(int, label_data1)

label_data2 = np.zeros(N)
label_data2[n1_data2:n1_data2 + n2_data2 - 1] = 1
label_data2[N - n3_data2:] = 2
label_data2 = map(int, label_data2)

def getParameters(mean1,mean2,mean3,cov1,cov2,cov3):
    mu=np.vstack((mean1, mean2, mean3))
    sigma= np.zeros((3, 2, 2))
    sigma[0], sigma[1], sigma[2] = cov1, cov2, cov3
    return mu,sigma

mean_X1,sigma_X1=getParameters(mean1_data1,mean2_data1,mean3_data1,cov1_data1,cov2_data1,cov3_data1)
mean_X2,sigma_X2=getParameters(mean1_data2,mean2_data2,mean3_data2,cov1_data2,cov2_data2,cov3_data2)