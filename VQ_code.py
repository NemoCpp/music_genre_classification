import pickle
from scipy.cluster.vq import vq,kmeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np


n_clusters = 200
centroids = {}

train_featues = pickle.load(open("all_train_features_normalized.pkl","r"))
print("train features loaded")

# codebook for hindi
X = []
for key in train_featues.keys():
	if(key[0] == "h"):
		for j in range(train_featues[key].shape[0]):
			X.append(train_featues[key][j])

X = np.array(X)
print(X.shape)
print("read hindi done")
train_featues = {}
hindi_kmeans = (MiniBatchKMeans(n_clusters = n_clusters,batch_size = 5000,verbose=0))
hindi_kmeans.fit(X)
centroids["hindi"] = hindi_kmeans.cluster_centers_
print("codebook hindi done")



train_featues = pickle.load(open("all_train_features_normalized.pkl","r"))
print("train features loaded")
# codebook for hindi
X = []
for key in train_featues.keys():
	if(key[0] == "t"):
		for j in range(train_featues[key].shape[0]):
			X.append(train_featues[key][j])

X = np.array(X)
print("read tamil done")
print(X.shape)
train_featues = {}
tamil_kmeans = (MiniBatchKMeans(n_clusters = n_clusters,batch_size = 5000,verbose=0))
tamil_kmeans.fit(X)
centroids["tamil"] = tamil_kmeans.cluster_centers_
print("codebook tamil done")

pickle.dump(centroids,open("VQ_codebook.pkl","wb"))

# test_featues = pickle.load("all_test_features_normalized.pkl")

