import numpy as np
import math
np.set_printoptions(suppress=True)#不用科学记数法输出数据

# 首先统计数据集的大小
f=open("C:\\Users\\Administrator\\Desktop\\Iris.txt")
n=0#用来记录数据的行数
for eachline in f:
    n=n+1
print("数据集的行数:")
print(n)
f.close()#关闭打开的文件
f=open("C:\\Users\\Administrator\\Desktop\\Iris.txt")
data=np.empty([n,4],dtype=float)#定义一个n行10列空矩阵
t=0
#按行读取文本文件，每行数据根据逗号"，"分开
for eachline in f:
    L=eachline.split(",")
    i=0
    for i in range(0,4):
        data[t,i]=float(L[i])
    t=t+1
f.close()
print("数据集如下：")
print(data)

#（1）使用输入空间中的核函数为数据计算居中和归一化的齐二次核矩阵K
K=np.empty([n,n],dtype=float)#定义一个n*n的核矩阵
Kmean=0#核矩阵的均值
Center_K=np.empty([n,n],dtype=float)
for i in range(0,n):
    for j in range(0,n):
        K[i,j]=round(math.pow(np.dot(data[i,:].T,data[j,:]),2),4)#计算核矩阵,并精确到小数点后四位
print("数据的齐二次核矩阵为：")
print(K)
for i in range(0,n):#求矩阵元素所有值总和
    for j in range(0,n):
        Kmean=Kmean+K[i,j]
Kmean=round((1/n)*np.sqrt(Kmean),4)#求均值并精确到小数点后四位
print("核矩阵的均值为：")
print(Kmean)
Center_K=K-Kmean#将矩阵居中化
Kmin=Center_K[0,0]
Kmax=0
for i in range(0,n):
    for j in range(0,n):
        if Kmax<K[i,j]:
            Kmax=K[i,j]
        if Kmin>K[i,j]:
            Kmin=K[i,j]
R=Kmax-Kmin#矩阵的极差
Center_Norm_K=(K-Kmin)/R#居中和中心化矩阵
print("居中和归一化之后的核矩阵")
print(Center_Norm_K)

#（2）使用齐次二次核将每个点x转换为特征空间，将这些点居中并对其进行归一化
Mapping_data=np.empty([n, 10], dtype=float)#特征空间映射数据集
Center_Mapping_data=np.empty([n, 10], dtype=float)#居中化之后的特征空间数据点集
C_N_M_data=np.empty([n, 10], dtype=float)#居中和归一化之后的特征空间数据点集
Mapping_X=np.empty([4, 4], dtype=float)#映射数据点
mean_X=np.empty([1,10],dtype=float)#数据点平均值
R_X=np.empty([1,10],dtype=float)#数据点极差
minX=np.empty([1,10],dtype=float)#数据点的极小值
maxX=0#数据点的极大值
for i in range(0,n):#对原数据点集进行特征空间的映射
    Mapping_X=np.outer(data[i, :], data[i, :])
    t=0
    for j in range(0,4):
        for k in range(j,4):
            Mapping_data[i, t]=Mapping_X[j, k]
            t=t+1
print("使用齐二次核每个点转换为特征空间中得到的点集为：")
# np.set_printoptions(threshold=1e6)#设置为全部输出，不省略输出数据
print(Mapping_data)
for i in range(0,10):#求出数据的平均值
    u=0
    for j in range(0,n):
        u= u + Mapping_data[j, i]
    mean_X[0,i]=(1/n)*u
for i in range(0,10):
    for j in range(0,n):
        Center_Mapping_data[j, i]= Mapping_data[j, i] - mean_X[0, i]
for i in range(0,10):
    minX[0,i]=Center_Mapping_data[0, i]
    for j in range(0,n):
        if maxX<Center_Mapping_data[j, i]:
            maxX=Center_Mapping_data[j, i]
        if minX[0,i]>Center_Mapping_data[j, i]:
            minX[0,i]=Center_Mapping_data[j, i]
    R_X[0,i]=maxX-minX[0,i]
for i in range(0,10):
    for j in range(0,n):
        C_N_M_data[j, i]= round((Center_Mapping_data[j, i] - minX[0, i]) / R_X[0, i], 4)#求数据在特征空间的映射
print("居中和归一化之后的映射在特征空间中点的数据集为：")
print(C_N_M_data)

#（3）验证特征空间居中和归一化点的成对点积是否产生相同的内核矩阵，通过核函数在输入空间中直接计算。
Mapping_Center_Norm_K=np.empty([n, n], dtype=float)
for i in range(0,n):
    for j in range(0,n):
        Mapping_Center_Norm_K[i, j]=math.pow(np.dot(C_N_M_data[i, :].T, C_N_M_data[j, :]), 2)#计算核矩阵
print("映射空间的居中化和归一化之后的核矩阵为：")
print(Mapping_Center_Norm_K)