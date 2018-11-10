
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

f=open("E:\\node2vec\\node2vec\\emb\\Les2.emb",'r')
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

ft=open("E:\\node2vec\\node2vec\\graph\\Les.edgelist",'r')
edges=ft.readlines()
colors=[]
g=nx.Graph()
for i in edges:
    edge=i.split()
    g.add_edge(edge[0],edge[1],weight=edge[2])

for i in g.nodes():
    colors.append(col[nodes_dic[i]])

nx.draw(g,with_labels=True,style='dashdot',pos=nx.spring_layout(g),node_color=colors)
plt.savefig("./ba4.png")           #使用matplotlib，保存网络图
plt.show()
# circular_layout：节点在一个圆环上均匀分布
# random_layout：节点随机分布
# shell_layout：节点在同心圆上分布
# spring_layout： 用Fruchterman-Reingold算法排列节点（这个算法我不了解，样子类似多中心放射状）
# spectral_layout：根据图的拉普拉斯特征向量排列节点？我也不是太明白






#TSNE非线性降维可视化
# tsne=TSNE()
# tsne.fit_transform(node2vecs)  #进行数据降维,降成两维
# #a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
# tsne=pd.DataFrame(tsne.embedding_) #转换数据格式
# # tsne[2]=labels
# plt.scatter(tsne[0], tsne[1], c=labels,s=labels*10)
# # plt.show()















