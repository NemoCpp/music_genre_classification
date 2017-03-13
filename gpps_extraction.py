import sidekit
import numpy as np
import pickle
from scipy.stats import multivariate_normal

ubm = pickle.load(open("ubm_64.pkl"))
print("ubm loaded")

train_features = pickle.load(open("all_train_features_normalized.pkl")) 
print("train loaded",len(train_features))
gpps_train = {}
for i in range(len(train_features.keys())):
	utt = train_features.keys()[i]
	feature_frames = train_features[utt]
	log_posteriors = ubm.compute_log_posterior_probabilities(feature_frames)
	posteriors = np.exp(log_posteriors)
	temp_mat = np.diag(1.0/posteriors.sum(axis=1))
	posteriors_normalized = np.dot(temp_mat,posteriors)
	gpps_vector = posteriors_normalized.mean(axis=0)
	gpps_train[utt] = gpps_vector

	if(np.mod(i,100)==0):
		print(i,"train")

pickle.dump(gpps_train,open("gpps_train.pkl","wb"))

test_features = pickle.load(open("all_test_features_normalized.pkl")) 
print("test loaded",len(test_features))
gpps_test = {}
for i in range(len(test_features.keys())):
	utt = test_features.keys()[i]
	feature_frames = test_features[utt]
	log_posteriors = ubm.compute_log_posterior_probabilities(feature_frames)
	posteriors = np.exp(log_posteriors)
	temp_mat = np.diag(1.0/posteriors.sum(axis=1))
	posteriors_normalized = np.dot(temp_mat,posteriors)
	gpps_vector = posteriors_normalized.mean(axis=0)
	gpps_test[utt] = gpps_vector
	if(np.mod(i,100)==0):
		print(i,"test")

pickle.dump(gpps_test,open("gpps_test.pkl","wb"))