import numpy as np
import math
np.set_printoptions(suppress=True)#不用科学记数法输出数据
from  sklearn.datasets import load_iris#数据

iris=load_iris()
dataset=np.empty([len(iris.data),5],dtype=float)
dataset[:,:4]=iris.data
dataset[:,-1]=iris.target

degree=4#维度设为全局变量

#求两个向量之间的距离
def distance(X1,Xi):
    dist=0#用来记录两点之间的距离
    for i in range(len(X1)):
        dist=dist+pow((X1[i]-Xi[i]),2)
    dist=math.sqrt(dist)
    return dist

#定义核函数
def kernelize(X,Xi,h,degree):
    dist=distance(X,Xi)
    kernel=(1/pow(2*np.pi,degree/2))*np.exp(-(dist*dist)/(2*h*h))
    return kernel

#均值漂移函数
def shiftPoint(center,points,h):
    axis=np.zeros([1,4])
    sumweight=0
    newcenter=np.zeros([1,4])#新的中心点
    for temp in points:
        weight=kernelize(center,temp,h,4)
        for i in range(0,4):
            axis[0,i]+=weight*temp[i]
        sumweight+=weight
    for j in range(0,4):
        axis[0,j]=axis[0,j]/sumweight
        newcenter[0,j]=axis[0,j]
    return newcenter

#定义寻找密度吸引子函数
def FindAttractor(X,D,h,si):
    t=0
    n=len(D)
    Xt=np.zeros([n,4])
    pointlist=[]
    Xt[t,:]=X
    for item in D:
        if distance(item,Xt[t,:])<=h:
            pointlist.append(item)
    Xt[t+1,:]=shiftPoint(X,pointlist,h)
    pointlist.clear()
    t=t+1
    while distance(Xt[t,:],Xt[t-1,:])>=si:
        # print(distance(Xt[t,:],Xt[t-1,:]))
        for item in D:
            if distance(item,Xt[t,:])<=h:
                pointlist.append(item)
        Xt[t+1]=shiftPoint(X,pointlist,h)
        pointlist.clear()
        t=t+1
#返回密度吸引子和漂移经过的点

    return Xt[t],Xt[1:t+1,:]


def DensityThreshold(X,points,h,degree):
    threshold=0
    for item in points:
        threshold=threshold+(1/len(points)*math.pow(h,degree))*kernelize(X,item,h,degree)
    return threshold

#定义DENCLUE函数
def Denclue(D:[],h:float,sigma:float,si:float):
    A=[]#定义一个用来存放密度吸引子的集合
    R={}#定义一个集合用来存放被吸引子吸引的点的集合
    C={}#定义一个两两密度可达的吸引子的极大可达子集组成的集合
    points=[]
    num_of_attractor=0
    need_shift=[True]*len(D)
    global degree
    w=0
    for i in range(0,len(D)):
        if not need_shift[i]:
            continue
        X_star,shiftpoints=FindAttractor(D[i,:],D,h,si)
        # print(DensityThreshold(X_star,D,h,degree))
        for x in range(0,len(D)):
            if distance(X_star,D[x,:])<=h:
                points.append(D[x,:])

        print(DensityThreshold(X_star,points,h,degree))
        if DensityThreshold(X_star,points,h,degree)>=sigma:
            A.append(X_star)
            R.setdefault(num_of_attractor,[])
            for item in shiftpoints:
                for j in range(0,len(D)):
                    if need_shift[j]:
                        if distance(item,D[j,:])<=h:#寻找魔都吸引子在均值漂移过程中路过的点，并标记为该密度吸引子的点
                            R.get(num_of_attractor).append(D[j,:])
                            need_shift[j]=False
            num_of_attractor+=1
        points.clear()

    #输出密度吸引子和密度吸引子所包含的点
    for i in range(0,len(A)):
        print("密度吸引子"+str(A[i]))
        print("密度吸引子吸引的点")
        print(R[i])

    #将密度相连的密度吸引子所吸引的数据点归并为一个个类
    t=0
    C_star=np.empty([len(A),1],dtype=int)
    for i in range(0,len(A)):
        C_star[i]=-1

    for k in range(0,len(A)):
        # print(distance(A[k],A[k+1]))
        if C_star[k]==-1:
            C_star[k]=t
            print("第"+str(t+1)+"类簇"+"所包含的密度吸引子有：")
            print(A[k])
            if k!=len(A):
                for i in range(k,len(A)-1):
                    if distance(A[k],A[i+1])<=h:#判断这两个密度吸引子是否直接密度可达
                    #如果这两个密度吸引子的距离小于窗口宽度，则表明这两个点是密度可达的，并记录
                        C_star[i+1]=C_star[k]
                        print(A[i+1])
            t=t+1


    num_of_class=0
    for i in range(0,len(A)):
        if C_star[i,0] not in C:
            num_of_class+=1
            C.setdefault(C_star[i,0],[])
            C.get(C_star[i,0]).append(R[i])
        else:
            C.get(C_star[i,0]).append(R[i])

    #统计每个类的个数
    class_element_number=[0]*num_of_class
    for i in range(0,len(A)):
        for j in range(0,num_of_class):
            if C_star[i]==j:
                class_element_number[j]+=len(R[i])
    print(C_star)
    #输出每个类簇的点
    for i in range(0,num_of_class):
        print("类簇"+str(i+1)+"的点有：")
        print(C[i])

    print("每个簇中数据集实例的个数为：")
    for i in range(0,num_of_class):
        print("类簇"+str(i+1)+":  "+str(class_element_number[i]))
    return

if __name__ == '__main__':
    # print("请设置聚类的窗口带宽：")
    # h=input()
    # print("请设密度吸引子的最小密度阈值：")
    # eta=input()
    # print("请设置迭代时两点的密度容差：")
    # si=input()
    Denclue(iris.data,1,0.02,0.0001)
