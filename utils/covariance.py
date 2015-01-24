import os
import numpy as np
from objectlm import HierarchicalObservation
from scipy.cluster.hierarchy import linkage, to_tree
from utils import get_adjacency_matrix, get_degree_matrix, assign_parents

def tfidf_covariance(texts, savepath):
    if not savepath.endswith("/"):
        savepath = savepath + "/"
    if os.path.exists(savepath + "__linkage_average.npy"):
        Z = np.load(savepath + "__linkage_average.npy")
    else:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(input = str,
                                 strip_accents = 'ascii',
                                 analyzer ='word',
                                 max_features=5000)
        y = vectorizer.fit_transform(" ".join(text) for text in texts)
        Z = linkage(y.todense(), method='average', metric='euclidean')
        np.save(savepath + "__linkage_average.npy", Z)

    if os.path.exists(savepath + "__covariance__.npy"):
        Cov = np.load(savepath + "__covariance__.npy")
        observables = HierarchicalObservation(Cov)
    else:
        root, nodes = to_tree(Z, rd=True)
        assign_parents(root)
        adj_mat = get_adjacency_matrix(nodes)
        deg_mat = get_degree_matrix(nodes)
        sigma = 5
        laplacian = np.diag(deg_mat) - adj_mat + 1/(sigma**2) * np.eye(len(deg_mat))
        Cov = np.linalg.inv(laplacian)[:len(texts), :len(texts)]
        np.save(savepath + "__covariance__.npy", Cov)
        observables = HierarchicalObservation(Cov)
    return observables

def objectlm_covariance(matrix, savepath, metric="cosine"):
    if not savepath.endswith("/"):
        savepath = savepath + "/"
    if os.path.exists(savepath + "__linkage_average.npy"):
        Z = np.load(savepath + "__linkage_average.npy")
    else:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        Z = linkage(matrix, method='average', metric=metric)
        np.save(savepath + "__linkage_average.npy", Z)
    if os.path.exists(savepath + "__covariance__.npy"):
        Cov = np.load(savepath + "__covariance__.npy")
        observables = HierarchicalObservation(Cov)
    else:
        root, nodes = to_tree(Z, rd=True)
        assign_parents(root)
        adj_mat = get_adjacency_matrix(nodes)
        deg_mat = get_degree_matrix(nodes)
        sigma = 5
        laplacian = np.diag(deg_mat) - adj_mat + 1/(sigma**2) * np.eye(len(deg_mat))
        Cov = np.linalg.inv(laplacian)[:matrix.shape[0], :matrix.shape[0]]
        np.save(savepath + "__covariance__.npy", Cov)
        observables = HierarchicalObservation(Cov)
    return observables