import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.stats import f, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#get the data
iris = datasets.load_iris()
data = iris.data
X = data[:int(len(data) * 0.3)]
test = data
#build the PCA model
n_components = 3
pca = PCA(n_components = n_components)
Ttrain = pca.fit_transform(X)

# pca = PCA(n_components='mle')
# pca.fit(X)
# n_components = pca.n_components_

#this is to get all the eigenvalues for calculation later
svd = PCA()
svd.fit(X)

# Level of significance
alpha = 0.05

# Q(or SPE(x)) statistics
P = pca.components_
# If not use the pca function, always need to mean center the data. pca.transform mean center the data automatically
x = test - pca.mean_
Q = x @ (np.identity(P.shape[1]) - P.transpose() @ P).dot(x.transpose())
Q = np.diagonal(Q)

# Q statistics threshold
residualEigen = svd.singular_values_[n_components:]

theta = [0] * 3  # [0, 0, 0]

for i in range(3):
  for j in range(len(residualEigen)):
    theta[i] += residualEigen[j] ** (i + 1)

h0 = 1 - 2 * theta[0] * theta[2] / 3 / (theta[1] ** 2)

ca = norm.ppf(1 - alpha, 0 , 1)

Q_alpha = theta[0] * (h0 * ca * np.sqrt(2 * theta[1]) / theta[0] + 1 + theta[1] * h0 * (h0 - 1) / theta[0] / theta[0]) ** (1 / h0)

# T2 statistics
diag = np.diag(pca.singular_values_)

Tr = pca.transform(test)

# T2 = trT diag(eigenvalue) tr
T2 = Tr @  diag .dot(Tr.transpose())
T2 = np.diagonal(T2)

# T2 statistics threshold
N, n = X.shape
a = n_components
T2_UCL = n * (N - 1) * (N + 1) / N / (N -n) * f.ppf(1 -alpha, n, N - n)
#T2_UCL = a * (n - a) * (n + 1) / n / (n -a) * f.ppf(1 -alpha, a, n - a)
# CI calculation
k = 0.5

CI = k * Q / Q_alpha + (1 - k) * T2 / T2_UCL


# plt results
plt.figure(1)
plt.plot(T2 / T2_UCL)
plt.title('T2 / T2_UCL')

plt.figure(2)
plt.plot(Q / Q_alpha)
plt.title('Q / Q_alpha')

plt.figure(3)
plt.plot(CI)
plt.title('CI')


plt.figure(4)
plt.plot(Tr[:, 0], Tr[:, 1], 'o', color='black')
plt.plot(Ttrain[:, 0], Ttrain[:, 1], 'o', color='red')
plt.title('CI')

fig = plt.figure(5, figsize=(8, 6))
y = iris.target
x_min, x_max = test[:, 0].min() - .5, test[:, 0].max() + .5
y_min, y_max = test[:, 1].min() - .5, test[:, 1].max() + .5
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(Tr[:, 0], Tr[:, 1], Tr[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()




