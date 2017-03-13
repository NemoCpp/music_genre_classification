from scipy.spatial import distance
import pickle
import numpy as np

n_clusters = 200

centroids = pickle.load(open("VQ_codebook.pkl","r"))
test_features = pickle.load(open("all_test_features_normalized.pkl","r"))

pred_labels = []
test_labels = []

# centroids_h = centroids["hindi"]
centroids_t = centroids["tamil"]
centroids_h = centroids["hindi"]

for key in test_features.keys():
	if(key[0] == "h"):
		test_labels.append(0)
	if(key[0] == "t"):
		test_labels.append(1)

	h_dist = 0
	t_dist = 0

	for j in range(test_features[key].shape[0]):

		point_id_t = distance.cdist([test_features[key][j]], centroids_t).argmin()
		t_dist = t_dist + distance.euclidean(test_features[key][j],centroids_t[point_id_t])

		point_id_h = distance.cdist([test_features[key][j]], centroids_h).argmin()
		h_dist = h_dist + distance.euclidean(test_features[key][j],centroids_h[point_id_h])

	# print(t_dist)
	if(h_dist < t_dist):
		pred_labels.append(0)
	else:
		pred_labels.append(1)


pred_labels = np.array(pred_labels)
test_labels = np.array(test_labels)

acc = float((pred_labels==test_labels).sum())/(pred_labels.shape[0])
print("Acc: ",acc)



# accuracy 77.91 with codebook size 500
# 
# Codebook size: Accuracy
# 500:	77.91
# 400:	77.08
# 300:	76.77
# 200:	75.57

a = [75.57, 76.77, 77.08, 77.91]
b = [200,300,400,500]

