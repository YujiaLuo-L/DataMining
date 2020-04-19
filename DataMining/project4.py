#encoding:utf-8
import numpy as np
import pandas as pd
from math import log
import json

np.set_printoptions(suppress=True)#不用科学记数法输出数据
from  sklearn.datasets import load_iris#数据

iris=load_iris()
dataset=np.empty([len(iris.data),5],dtype=float)
dataset[:,:4]=iris.data
dataset[:,-1]=iris.target

#计算信息的熵
def Ent(D):
    n=len(D)
    labelCount={}
    for item in dataset:
        currentLabel=item[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    #计算数据集D的熵
    EntD=0
    p=0
    for key in labelCount:
        p=labelCount[key]/n
        EntD=-p*log(p,2)+EntD
    return EntD

#计算信息增益
def Gain(D,DY,DN):
    gain=Ent(D)-(len(DY)/len(D))*Ent(DY)-(len(DN)/len(D))*Ent(DN)
    return gain

def Evaluate__Numeric_Attribute(D,X):
    #对数据进行排序
    (n,m)=D.shape#获取数据集的行数和列数
    D_sort=np.zeros([n,m])
    X_sort=np.zeros([n,1])

    #简单选择排序
    for j in range(0,n):
        count=j
        for i in range(0,n):
            if X[count]>X[i]:
                count=i
        D_sort[j,:]=D[count,:]
        X_sort[j]=X[count]
        X[count]=X[count]+100
    for j in range(0,n):
        X[j]=X[j]-100
    M=[]#存放所有的切分点c
    k=0#首先统计类的个数
    classListCount={}#统计每个种类的标签和种类的总数
    for item in D:
        currentLabel=item[-1]
        if currentLabel not in classListCount:
            classListCount[currentLabel]=0
            k=k+1
        classListCount[currentLabel]+=1

    ni=np.empty([1,k])#用来统计当前划分点左边每个种类的数目
    for i in range(0,k):
        ni[0,i]=0#初始化每一类点的个数为0

    Nvi=np.zeros([n,k],dtype=int)#用来存放分切点左边的数据的分类和数目
    w=0#用来表示切分点的数目
    #寻找切分点，属性中连续的两个不同值之间切分，切分点的大小取两个值的平均值
    for j in range(0,n-1):
        for r in range(0,k):
            if int(D_sort[j,-1])==r:
                ni[0,r]=ni[0,r]+1
        if X_sort[j+1]!=X_sort[j]:
                v=(X_sort[j+1]+X_sort[j])/2#两个连续不同值的中点作为数据的切分点
                M.append(v)#存储切分点值
                for i in range(0,k):
                    Nvi[w,i]=ni[0,i]
                w=w+1

    #最后一个点不作为分裂点
    for i in range(0,k):
        if D_sort[n-1,-1]==i:
            ni[0,i]=ni[0,i]+1

#分切点
    v_star=0
    score_star=0
    num_v=0
    for v in M:
        PDY=0
        PDN=0
        sum=0
        for i in range(0,k):
           sum=sum+Nvi[num_v][i]
        for j in range(0,k):
            PDY+=Nvi[num_v,i]/sum
            if j in classListCount:
                PDN+=(classListCount[j]-Nvi[num_v,i])/(n-sum)
        #将数据集划分为左右数据集
        num_v+=1

        DY=np.zeros([sum,m])
        DN=np.zeros([n-sum,m])
        for x in range(0,sum):
            DY[x,:]=D_sort[x,:]
        for y in range(sum,n):
            DN[y-sum,:]=D_sort[sum,:]
        score=Gain(D,DY,DN)
        if score>score_star:
            score_star=score
            v_star=v
    return v_star,score_star
number=0
#创建决策树，dataset数据集，labels数据标签
def createDecesionTree(dataset,eta,pi):
    labels=["sepal length", "sepal width", "petal length", "petal width"]
    n=len(dataset)
    m=len(labels)
    labelCount={}
    currentLabel=None
    for item in dataset:
        currentLabel=item[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    #求纯度
    #纯度等于数据集中的最大的类占的概率
    p=0
    for key in labelCount:
        prob=float(labelCount[key])/n
        if p<prob:
            p=prob
            currentLabel=key
    #当数据集的长度小于5，或者纯度大于0.95时，可以停止划分并返回类标签
    if n<=eta or p>=pi:
        #返回数据的分类，最大分类标签，分类分精度，个分类的大小
        return '分类'+str(currentLabel)+' 纯度'+str(p)+' 数据集大小：'+str(n)

    #如果不满足结束划分的条件
    #寻找最佳分裂点对数据进行划分

    score_star=0#用来表示最大信息增益
    v_star=0
    bestSplitFeat=0
    #遍历每一个特征求最佳的划分点和信息增益
    for i in range(0,m):
        v,score=Evaluate__Numeric_Attribute(dataset,dataset[:,i])#切分点和信息增益
        if score>score_star:
            bestSplitFeat=i
            score_star=score
            v_star=v
    global number
    number+=1
    bestFeatLabel=labels[bestSplitFeat]+'<'+str(v_star)
    Tree={bestFeatLabel:{}}

    retDatasetLeft=[]#存储左子树的数据
    retDatasetRight=[]#存储右子树的数据
    #获取左子树数据集
    for featVec in dataset:
        if float(featVec[bestSplitFeat]) < v_star:
            retDatasetLeft.append(featVec)
    #获取右子树数据集
    for featVec in dataset:
        if float(featVec[bestSplitFeat]) >= v_star:
            retDatasetRight.append(featVec)
    retDatasetRight=np.array(retDatasetRight)
    retDatasetLeft=np.array(retDatasetLeft)
    #分别构建左子树和右子树
    valueLeft='Yes'+'信息增益'+str(score_star)#如果满足切分条件则返回信息增益
    Tree[bestFeatLabel][valueLeft] = createDecesionTree(retDatasetLeft,eta,pi)
    # 构建右子树
    valueRight='No'
    Tree[bestFeatLabel][valueRight] = createDecesionTree(retDatasetRight,eta,pi)
    return Tree

if __name__ == '__main__':
    number=0
    tree = createDecesionTree(dataset,5,0.95)
    #print("决策树为：")
    #print(tree)
    print("鸢尾花数据为：")
    print(dataset)
    print("决策树的分类结点大致有：")
    print(str(number)+'个')
    f=open("C:\\Users\\Administrator\\Desktop\\decisionTree.txt","w+")
    f.write(json.dumps(tree))#利用json的dumps将决策树嵌套字典写入文件中
    f.write('\n')
    f.close()







