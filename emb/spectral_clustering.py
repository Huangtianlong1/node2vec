import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import  KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

k=16  #降维后的向量长度
ft=open("E:\\node2vec\\node2vec\\graph\\Les.edgelist",'r')
edges=ft.readlines()
g=nx.Graph()
for i in edges:
    edge=i.split()
    g.add_edge(int(edge[0]),int(edge[1]),weight=int(edge[2]))
print(g.edges())
#标准化的拉普拉斯矩阵
L=nx.adjacency_matrix(g)
# print(L.todense())
L1=np.array(L.todense())
#聚类
# spectral=SpectralClustering(n_clusters=k,affinity='nearest_neighbors',n_neighbors=4,eigen_solver='arpack',n_jobs=20)
# pre=spectral.fit(L1)
L_new=nx.normalized_laplacian_matrix(g)
a,b=np.linalg.eig(np.array(L_new.todense()))
print(a)
print(b)
eig_vec_dic={}
eig_vec_dic=dict(zip(a,b))
result=list(sorted(eig_vec_dic.items(),key=lambda x:x[0]))
coll=[]
for i in range(k):
    coll.append(result[i][1])
#coll中的每一列存放的是一个节点降维后的向量
coll1=np.array(coll)
print(coll1.shape)
A=coll1.transpose()  #降维后的样本

#聚类
kmeans=KMeans(n_clusters=6)
predic=kmeans.fit(A)
#获取聚类标签
labels=predic.labels_
#网络可视化
nodes=range(1,78)  #节点的标签
nodes_dic=dict(zip(nodes,labels))
col={0:'y',1:'r',2:'b',3:'g',4:'m',5:'w'}
colors=[]
for i in g.nodes():
    colors.append(col[nodes_dic[i]])

nx.draw(g,with_labels=True,style='dashdot',pos=nx.spring_layout(g),node_color=colors)
plt.savefig("./ba3.png")           #使用matplotlib，保存网络图
plt.show()





