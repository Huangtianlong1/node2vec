'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import src.node2vec as node2vec
from gensim.models import Word2Vec
import graph.graph1 as gh
import emb.line1 as line1
import emb.classify as clfy
from sklearn.linear_model import LogisticRegression

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='E:\\node2vec\\node2vec\\graph\\blogCatalog\\bc_edgelist.txt',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='E:\\node2vec\\node2vec\\emb\\blogCatalog.emb',
						help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
					  help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--epochs',type=int,default=5,
						help='The training epochs of LINE and GCN')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--label-file', dest='label',nargs='?',default='E:\\node2vec\\node2vec\\graph\\blogCatalog\\bc_labels.txt',
						help='The file of node label')

	parser.add_argument('--feature-file',nargs='?', default=False,
						help='The file of node features')

	parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
						help='Input graph format')

	parser.add_argument('--negative-ratio', default=5, type=int,
						help='the negative ratio of LINE')

	# parser.add_argument('--weighted', action='store_true',
	# 					help='Treat graph as weighted')

	parser.add_argument('--clf-ratio', default=0.5, type=float,
						help='The ratio of training data in the classification')

	parser.add_argument('--order', default=3, type=int,
						help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')

	parser.add_argument('--no-auto-save', action='store_true',
						help='no save the best embeddings when training LINE')

	parser.add_argument('--dropout', default=0.5, type=float,
						help='Dropout rate (1 - keep probability)')

	parser.add_argument('--weight-decay', type=float, default=5e-4,
						help='Weight for L2 loss on embedding matrix')

	parser.add_argument('--hidden', default=16, type=int,
						help='Number of units in hidden layer 1')

	parser.add_argument('--kstep', default=4, type=int,
						help='Use k-step transition probability matrix')

	parser.add_argument('--lamb', default=0.2, type=float,
						help='lambda is a hyperparameter in TADW')

	parser.add_argument('--lr', default=0.01, type=float,
						help='learning rate')

	parser.add_argument('--alpha', default=1e-6, type=float,
						help='alhpa is a hyperparameter in SDNE')

	parser.add_argument('--beta', default=5., type=float,
						help='beta is a hyperparameter in SDNE')

	parser.add_argument('--nu1', default=1e-5, type=float,
						help='nu1 is a hyperparameter in SDNE')

	parser.add_argument('--nu2', default=1e-4, type=float,
						help='nu2 is a hyperparameter in SDNE')

	parser.add_argument('--bs', default=200, type=int,
						help='batch size of SDNE')

	parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
						help='a list of numbers of the neuron at each encoder layer, the last number is the ''dimension of the output node representation')

	return parser.parse_args()


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y



def read_graph():
	'''
	Reads the input network in networkx....
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()


	if args.label:
		model = line1.LINE(G, epoch=args.epochs, rep_size=args.dimensions, order=args.order,
					   label_file=args.label, clf_ratio=args.clf_ratio)
	else:
		model = line1.LINE(G, epoch=args.epochs,
					   rep_size=args.dimensions, order=args.order)

	print("Saving embeddings...")
	model.save_embeddings(args.output)

	vectors = model.vectors
	X, Y = read_node_label(args.label) #dest给变量起名字
	print("Training classifier using {:.2f}% nodes...".format(args.clf_ratio * 100))
	clf = clfy.Classifier(vectors=vectors, clf=LogisticRegression())
	clf.split_train_evaluate(X, Y, args.clf_ratio)

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]

	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)



	
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	# nx_G = read_graph()  #node2vec使用


	#使用LINE
	nx_G=read_graph()


#    nx_G = read_graph()  #node2vec使用
#使用node2vec
	# G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	# G.preprocess_transition_probs()
	# walks = G.simulate_walks(args.num_walks, args.walk_length)
	# learn_embeddings(walks)


if __name__ == "__main__":
	args = parse_args()
	main(args)
