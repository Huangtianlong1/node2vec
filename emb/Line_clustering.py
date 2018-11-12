from sklearn.cluster import  KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import  networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import argparse
f=open("E:\\node2vec\\node2vec\\emb\\blogCatalog.emb",'r')
alls=f.readlines()
node_num,feature_num=alls[0].split(' ')
nodes=[i.split(' ')[0] for i in alls[1:]]
node2vec_list=[i.split(' ')[1:] for i in alls[1:]]
node2vecs=np.array(node2vec_list)
meandistortions=[]
k=6
kmeans=KMeans(n_clusters=k)
#聚类
predic=kmeans.fit(node2vecs)
#获取聚类标签
labels=predic.labels_
# print(len(labels))
#获取聚类中心
centers=predic.cluster_centers_
node2vecs1=node2vec_list+centers.tolist()
# node2vecs1=node2vecs.tolist()+centers
print(node2vecs1)
#PCA线性降维可视化
#调用sklearn中的PCA，其中主成分有5列
# pca_sk = PCA(n_components=2)
# print(labels)
# #利用PCA进行降维，数据存在newMat中
# newMat = pca_sk.fit_transform(node2vecs1)
#
# #可视化降维后的结果
# plt.scatter(newMat[:len(newMat)-k,0], newMat[:len(newMat)-k,1], c=labels,s=labels*10)  #c参数颜色，s参数大小
# #标记聚类中心
# plt.scatter(newMat[len(newMat)-k:,0],newMat[len(newMat)-k:,1],c=[5,6,7,8],s=[80,100,120,140])
# # plt.show()


#网络可视化
nodes_dic=dict(zip(nodes,labels))
col={0:'y',1:'r',2:'b',3:'g',4:'m',5:'w'}  #3:'g',4:'m',5:'w'

ft=open("E:\\node2vec\\node2vec\\graph\\blogCatalog\\bc_edgelist.txt",'r')
edges=ft.readlines()
colors=[]
g=nx.Graph()
for i in edges:
    edge=i.split()
    g.add_edge(edge[0],edge[1],weight=1)

for i in g.nodes():
    colors.append(col[nodes_dic[i]])

nx.draw(g,with_labels=True,style='dashdot',pos=nx.spring_layout(g),node_color=colors)
plt.savefig("./ba5.png")           #使用matplotlib，保存网络图
plt.show()