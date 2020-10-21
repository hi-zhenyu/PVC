import numpy as np
import sklearn.metrics as metrics
import torch
from munkres import Munkres
import warnings
warnings.filterwarnings('ignore')

def normalize(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x

def nan_check(x, name):
    if type(x) == torch.Tensor:
        assert not torch.isnan(x).any(), 'Nan found in ' + name
        assert not torch.isinf(x).any(), 'inf found in ' + name
    elif type(x) == np.ndarray:
        assert not np.isnan(x).any(), 'Nan found in ' + name
        assert not np.isinf(x).any(), 'inf found in ' + name

def namestr(obj, namespace=locals()):
    print(namespace)
    for name in namespace:
        if namespace[name] is obj:
            return name

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments)!=0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def kmeans(features, Y):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=len(np.unique(Y)), n_init=20)
    km_pred = km.fit_predict(features)
    y_preds = get_y_preds(Y, km_pred, len(np.unique(Y)))
    scores = metric(Y, km_pred, len(np.unique(Y)))

    return y_preds, scores 

def metric(y_true, y_pred, n_clusters, verbose=True, decimals = 4):
    
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = float(np.round(nmi, decimals))
    
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred_ajusted)
    accuracy = float(np.round(accuracy, decimals))

    # F-score
    f_score = metrics.f1_score(y_true, y_pred_ajusted, average='weighted')
    f_score = float(np.round(f_score, decimals))
    
    if verbose:
        print('ACC:', accuracy, 'NMI:', nmi, 'F-mea:', f_score)
    return dict({'ACC': accuracy, 'NMI': nmi, 'F-mea':f_score})


def euclidean_dist(x, y, root = False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist