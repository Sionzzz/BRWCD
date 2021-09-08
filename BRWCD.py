# encoding=utf8

import networkx as nx
import numpy as np
import collections

import scipy.sparse
import scipy.sparse as sp

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

import argparse
import time

import theano
from theano import tensor


def svd_deepwalk_matrix(X, dim):
    u, s, v = scipy.sparse.linalg.svds(X, dim, return_singular_vectors="u")
    return scipy.sparse.diags(np.sqrt(s)).dot(u.T).T


def direct_compute_deepwalk_matrix(T, window, Neg_minus):

    n = T.shape[0]
    ne1 = Neg_minus.copy()
    ne2 = Neg_minus.copy()

    Tr = np.zeros_like(T)
    Tr_power = sp.identity(n)

    for i in range(window):
        Tr_power = Tr_power.dot(T)
        Tr += Tr_power

    TF1 = Tr.dot(ne1)
    TF2 = ne2.dot(Tr.T)
    S = 0.5 * (TF1 + TF2)
    S *= n / window

    m = tensor.matrix()
    f = theano.function([m], tensor.log(tensor.maximum(m, 1)))
    Y = f(S.todense().astype(theano.config.floatX))

    return sp.csr_matrix(Y)


class BiasedWalk():
    def __init__(self, graph_file, dimension, feat_file, attWeight):
        self.graph = graph_file
        # print("dimension =", dimension)
        self.dimension = dimension

        self.G = nx.read_edgelist(self.graph, nodetype=int, create_using=nx.DiGraph())
        self.G = self.G.to_undirected()
        self.G.remove_edges_from(self.G.selfloop_edges())

        maxIndex = max([node for node in self.G.nodes()])
        self.node_number = maxIndex + 1
        # print("node_num =", self.node_number)

        feat = collections.defaultdict(list)
        with open(feat_file) as file:
            line = file.readlines()[0].strip().split()
            feat_number = len(line) - 1
        # print("feat_num =", feat_number)

        featList = [[] for i in range(feat_number)]

        self.feat_type = 0
        feat_edge = 0
        for line in open(feat_file):
            line = line.strip().split()
            n = int(line[0])
            del line[0]
            feat[n] = list(map(float, line))
            for d, f in enumerate(feat[n]):
                if f != 0:
                    featList[d].append(n)
                    feat_edge += 1
                    if f != int(f) and self.feat_type == 0:
                        self.feat_type = 1

        self.shape_mat = self.node_number + feat_number

        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))
        matrix1 = scipy.sparse.lil_matrix((self.node_number, feat_number))
        matrix2 = scipy.sparse.lil_matrix((self.shape_mat, self.shape_mat))
        matrix_conn = scipy.sparse.lil_matrix((self.node_number, feat_number))

        sparsity = min(float(self.node_number) / self.G.number_of_edges(), 1)

        strWeight = 1-attWeight
        for e in self.G.edges():
            u = int(e[0])
            v = int(e[1])
            if u != v:
                matrix0[u, v] = 1
                matrix0[v, u] = 1

                matrix2[u, v] = strWeight
                matrix2[v, u] = strWeight

        for d in range(feat_number):
            nodeList = featList[d]
            for i in range(len(nodeList)):
                vi = nodeList[i]
                nes = set(self.G.neighbors(vi)) if vi in self.G.nodes else set()
                nodeSet = set(nodeList)

                conn = 1 + len(nes.intersection(nodeSet))
                connStr = conn / len(nes) if len(nes) != 0 else 0
                connAtt = conn / len(nodeSet) if len(nodeSet) != 0 else 0
                connSA = sparsity * connStr + (1 - sparsity) * connAtt

                matrix_conn[vi, d] = connSA * (feat[vi][d] ** 1)

                matrix1[vi, d] = feat[vi][d]
                matrix2[vi, self.node_number + d] = attWeight * feat[vi][d]
                matrix2[self.node_number + d, vi] = attWeight * feat[vi][d]

        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        self.matrix1 = scipy.sparse.csr_matrix(matrix1)
        self.matrix2 = scipy.sparse.csr_matrix(matrix2)
        self.matrix_conn = scipy.sparse.csr_matrix(matrix_conn)

    def get_embedding_rand(self, matrix):
        t1 = time.time()
        smat = scipy.sparse.csc_matrix(matrix)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    def trans_mat(self, matA, matB, matC, matConn, attWeight, alpha, step):
        strWeight = 1 - attWeight

        TranB = preprocessing.normalize(matConn, "l1")
        TranB = attWeight * TranB
        rowsB = TranB.nonzero()[0]
        colsB = TranB.nonzero()[1]

        NormF = preprocessing.normalize(matC, "l1")
        F = np.array(NormF.sum(axis=0))[0]
        F_zeros = np.where(F == 0)
        F_minus = F ** -1
        F[F_zeros] = 0
        diag_F_minus = scipy.sparse.diags(F_minus, format="csr")

        InfosForAtt = np.array(preprocessing.normalize(matB, "l1").sum(axis=0))[0]
        infoAtt_zeros = np.where(InfosForAtt == 0)
        InfosForAtt_minus = InfosForAtt ** -1
        InfosForAtt_minus[infoAtt_zeros] = 0

        Norm_A = preprocessing.normalize(matA, "l1")
        infos = np.array(Norm_A.sum(axis=0))[0]
        info_zeros = np.where(infos == 0)
        infos_minus = infos ** -1
        infos_minus[info_zeros] = 0

        infos = 1 / (1 + np.exp(-alpha * infos_minus))
        diag_infos_minus_A = scipy.sparse.diags(infos, format="csr")

        TranC = scipy.sparse.lil_matrix((matB.shape[1], matA.shape[0]))
        for i in range(len(rowsB)):
            row = rowsB[i]
            col = colsB[i]
            TranC[col, row] = (attWeight * InfosForAtt_minus[col] + strWeight * infos_minus[row])
            TranC[col, row] = 1 / (1 + np.exp(-alpha * TranC[col, row]))
        TranC = TranC.tocsr()
        TranC = preprocessing.normalize(TranC, "l1")

        TranA = matA.dot(diag_infos_minus_A)
        TranA = preprocessing.normalize(TranA, "l1")
        TranA = strWeight * TranA

        zeros = scipy.sparse.lil_matrix((matB.shape[1], matB.shape[1]))
        Tran_AB = scipy.sparse.hstack((TranA, TranB))
        Tran_C0 = scipy.sparse.hstack((TranC, zeros))
        Tran = scipy.sparse.vstack((Tran_AB, Tran_C0))

        Tran = Tran.tocsr()
        F_zzz = direct_compute_deepwalk_matrix(T=Tran, window=step, Neg_minus=diag_F_minus)

        features_matrix = self.get_embedding_rand(F_zzz)

        return features_matrix

    def save_embedding(self, emb_file, features):
        f_emb = open(emb_file, 'w')
        f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
        for i in range(len(features)):
            s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
            f_emb.write(s + "\n")
        f_emb.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiasedWalk.")
    parser.add_argument('-graph', nargs='?', default='cora/network.txt',
                        help='Graph path')
    parser.add_argument('-output', nargs='?', default='emb/cora.emb',
                        help='Output path of sparse embeddings')
    parser.add_argument('-dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('-attributes', nargs='?', default='cora/features.txt',
                        help='Attributes of nodes')
    parser.add_argument('-step', type=int, default=5,
                        help='Step of recursion. Default is 5.')
    parser.add_argument('-alpha', type=int, default=15,
                        help='Step of recursion. Default is 5.')
    parser.add_argument('-wt', type=int, default=0.5,
                        help='Step of recursion. Default is 5.')
    return parser.parse_args()


def main():
    args = parse_args()

    model = BiasedWalk(args.graph, args.dimension, args.attributes, args.wt)
    features_matrix = \
        model.trans_mat(model.matrix0, model.matrix1, model.matrix2, model.matrix_conn, args.wt, args.alpha, args.step)
    model.save_embedding(args.output, features_matrix)


if __name__ == '__main__':
    main()