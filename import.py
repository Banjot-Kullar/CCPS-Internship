import scipy.io
import pandas as pd
mat = scipy.io.loadmat('filename.mat')
data = mat['X']
labels = mat['Y'].ravel()
X = pd.DataFrame(data.T, columns=[f'Instance_{i}' for i in range(data.shape[0])])
y = pd.DataFrame(labels, columns=['label'])
X=X.T
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')