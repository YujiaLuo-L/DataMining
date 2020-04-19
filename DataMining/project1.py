#-*- coding:utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
import string
import math
np.set_printoptions(suppress=True)#不用科学记数法输出数据
#读取数据
# 首先统计magic04数据集的行数
f=open("C:\\Users\\Administrator\\Desktop\\magic04.txt")
n=0#用来记录数据的
for eachline in f:
    n=n+1
print("数据集的行数:")
print(n)
f.close()#关闭打开的文件
f=open("C:\\Users\\Administrator\\Desktop\\magic04.data")
data=np.empty([n,10],dtype=float)#定义一个n行10列空矩阵
t=0
#按行读取文本文件，每行数据根据逗号"，"分开
for eachline in f:
    L=eachline.split(",")
    i=0
    for i in range(0,10):
        data[t,i]=float(L[i])
    t=t+1
f.close()
print("数据集如下：")
print(data)

#（1）求数据的均值向量
#每一列表示一个属性，向量均值则求每一列数值的平均值
M=np.empty([1,10],dtype=float)#行向量
t=0
for i in range(0,10):
    for j in range(0,n):#求每列数据的总和
        t=t+data[j,i]
    M[0,i]=round(t/n,4)#数据的平均值
print("数据集的均值向量为：")
print(M)


#（2）计算样本协方差矩阵作为中心点数据矩阵各列之间的内积
#计算样本的中心矩阵
Z=np.empty([n,10],dtype=float)#定义一个n行10列空矩阵
for i in range(0,10):
    for j in range(0,n):
        Z[j,i]=round(data[j,i]-M[0,i],4)
print("数据集中心化矩阵为：")
print(Z)
 #求协方差矩阵
CovIn=np.empty([10,10],dtype=float)
CovIn=(1/n)*(np.dot(Z[:].T,Z[:]))
print("样本的协方差矩阵(1)为：")
print(CovIn)

#（3）计算样本的协方差矩阵作为中心数据点之间的外部乘积
CovOut=np.empty([10,10],dtype=float)
Z1=np.empty([10,n],dtype=float)
Z1=Z.T#中心矩阵的转置矩阵
for i in range(0,n):
    CovOut=CovOut+np.outer(Z1[:,i],Z[i])
CovOut=(1/n)*CovOut
print("样本的协方差矩阵(2)为：")
print(CovOut)

#（4）计算中心属性向量之间角度的余弦值，计算属性1和属性2之间的相关性，绘制这两个属性之间的散点图
#计算中心属性向量之间的角度的余弦值矩阵（为主对角线对称矩阵）
cosTheta=np.empty([10,10],dtype=float)
for i in range(0,10):
    for j in range(i,10):
        v1=v2=v3=0#v1用来记录两个属性向量的点乘值，v2、v3用来表示中心属性i和j的模
        for k in range(0,n):
            v1=v1+Z[k,i]*Z[k,j]#求向量之间的成积
            v2=v2+Z[k,i]**2
            v3=v3+Z[k,j]**2
        v2=v2**0.5#求模，开平方
        v3=v3**0.5#求模，开平方
        cosTheta[i,j]=cosTheta[j,i]=round(v1/(v3*v2),4)#求余弦值，并精确到小数点后四位
print("中心属性的余弦值矩阵为：")
print(cosTheta)
#计算属性1和属性2的相关系数
std1=0#属性1的标准差
std2=0#属性2的标准差
corr12=0#属性1和2的相关系数
for i in range(0,n):
    std1=std1+(data[i,0]-M[0,0])**2
    std2=std2+(data[i,1]-M[0,1])**2
std1=((1/n)*std1)**0.5
std2=((1/n)*std2)**0.5
corr12=round(CovIn[0,1]/(std1*std2),4)#求相关系数并精确到小数点后四位
print("属性1和属性2的相关系数为："+str(corr12))
#绘制属性1和属性2的散点图
plt.scatter(data[:,0],data[:,1],marker='o',c='g')#画数属性1和属性2的散点图
plt.title("Scatter for Attribute 1 and 2")
plt.xlabel("Attribute 1")
plt.ylabel("Attribute 2")
plt.show();

#（5）假设属性1是正态分布的，绘制其概率密度函数。
x=np.linspace(M[0,0]-200,M[0,0]+200)#横轴的范围
y=np.exp(-(x-M[0,0])**2/(2*std1**2))/(math.sqrt(2*math.pi)*std1)#属性1 的正态分布函数
plt.plot(x,y,"g-",linewidth=2)
plt.title("Probability Density Function of Attribute 1 ")#函数的标题
plt.grid(True)
plt.show()

#（6）哪个属性方差最大，哪个属性方差最大？打印这些值
Var=np.empty([1,10],dtype=float)
v4=0#用来记录属性中每个值和属性均值相减后的平方和
t=t1=t2=t3=0#t、t1和t2、t3分别用来记录方差最大、最小的属性的方差值和属性号
for i in range(0,10):#计算样本属性的方差
    for j in range(0,n):
        v4=v4+(data[j,i]-M[0,i])**2
    v4=1/n*v4
    Var[0,i]=v4
t2=Var[0,i]
for k in range(0,10):
    if Var[0,k]>t:#寻找属性中方差最大的值
        t=round(Var[0,k],4)#若当前方差数值比t大，则赋值给t并只取到小数点后四位
        t1=k+1#记录属性号
    if t2>Var[0,k]:#寻找属性中方差最小的值
        t2=round(Var[0,k],4)#若当前方差数值比t2小，则赋值给t2并只取到小数点后四位
        t3=k+1
print("数据中每个属性的方差为：")
print(Var)
print("数据中，第"+str(t1)+"个属性的方差最大，为："+str(t))
print("数据中，第"+str(t3)+"个属性的方差最小，为："+str(t2))


 #（7）哪个属性对协方差最大，哪个属性对协方差最小？打印这些值
p=p1=p2=p3=0#p、p1和p2、p3分别用来记录写方差最大的属性对的协方差和属性对号
p=p2=CovIn[0,0]#将这两个数都初始化为第一个属性对的协方差值
for i in range(0,10):
    for j in range(i,10):
        if p<CovIn[i,j]:#求协方差最大的属性对和协方差的值
            p=round(CovIn[i,j],4)
            p1=str(i+1)+str(j+1)#记录属性对号，由于矩阵的行列以0开始而属性以1开始数，因此需要加1
        if p2>CovIn[i,j]:#求协方差最小的属性对和协方差的值
            p2=round(CovIn[i,j],4)
            p3=str(i+1)+str(j+1)
print("属性对"+str(p1)+"的协方差最大，为："+str(p))
print("属性对"+str(p3)+"的协方差最小，为："+str(p2))