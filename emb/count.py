f=open("E:\\node2vec\\node2vec\\graph\\blogCatalog\\bc_labels.txt",'r')
alls=f.readlines()
lis=[]
for i in alls:
    line=list(map(int,i.split()[1:]))
    print(line)
    lis.extend(line[1:])
#38个标签
print(set(lis))

