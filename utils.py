import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs

import matplotlib as mpl
import matplotlib.pyplot as plt

def gen_data(m, seed=1, scale=0.7):
    c=0
    x_train = np.linspace(0,49,m)
    np.random.seed(seed=seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def gen_blob():
    classes = 6
    m = 800
    std = 0.4
    centers = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]])
    X, y = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=2, n_features=2)
    return (X, y, centers, classes, std)

class LinModel:
    def __init__(self, degree=10, regularization=False, lambda_=0):
        if regularization:
            self.linear_model = Ridge(alpha=lambda_)
        else:
            self.linear_model = LinearRegression()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    def fit(self, X_train, y_train):
        X_train_mapped = self.poly.fit_transform(X_train.reshape(-1,1))
        X_train_mapped_scaled = self.scaler.fit_transform(X_train_mapped)
        self.linear_model.fit(X_train_mapped_scaled, y_train)
    
    def predict(self, X):
        X_mapped = self.poly.transform(X.reshape(-1,1))
        X_mapped_scaled = self.scaler.transform(X_mapped)
        ypred = self.linear_model.predict(X_mapped_scaled)
        return(ypred)
    
    def mse(self, y, ypred):
        err = mean_squared_error(y, ypred)/2
        return(err)

def plot_cat_decision_boundary(ax, X,predict , class_labels=None, legend=False, vector=True, color='g', lw = 1):
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    h = max(x_max-x_min, y_max-y_min)/200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]
    
    if vector:
        Z = predict(points)
    else:
        Z = np.zeros((len(points),))
        for i in range(len(points)):
            Z[i] = predict(points[i].reshape(1,2))
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, colors=color, linewidths=lw) 
    ax.axis('tight') 

def plt_mc_data(ax, X, y, classes,  class_labels=None, map=plt.cm.Paired, legend=False, size=50, m='o'):
    normy = mpl.colors.Normalize(vmin=0, vmax=classes)
    for i in range(classes):
        idx = np.where(y == i)
        label = class_labels[i] if class_labels else "c{}".format(i)
        ax.scatter(X[idx, 0], X[idx, 1],  marker=m,
                    color=map(normy(i)),
                    s=size, label=label)
    if legend: ax.legend(loc='lower right')
    ax.axis('equal')

def eval_cat_err(y, ypred):
    m = len(y)
    incorrect = 0
    for i in range(m):
        if ypred[i] != y[i]:
            incorrect += 1
    caterr = incorrect/m
    return(caterr)