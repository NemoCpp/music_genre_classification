import pickle
import numpy as np 
import os

feature_dic = pickle.load(open("mfcc_features.pkl","r"))
print("featuers loaded")
class_ids = os.listdir("genres")
print(class_ids)
np.save("class_ids.npy",class_ids)

train_names = []
test_names = []
train_labels = []
test_labels = []
train_features = []
test_features = []

paths_list = feature_dic.keys()
import random
random.seed(0)
random.shuffle(paths_list)
n_train = int(len(paths_list)*0.8)

train_names = paths_list[0:n_train]
test_names = paths_list[n_train:]

for i in train_names:
	tt = feature_dic[i][0:2998]
	train_features.append(tt)
	j = i.split(".")[0]
	train_labels.append(class_ids.index(j))

for i in test_names:
	tt = feature_dic[i][0:2998]
	test_features.append(tt)
	j = i.split(".")[0]
	test_labels.append(class_ids.index(j))


train_features = np.array(train_features)
test_features = np.array(test_features)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_features.shape)
print(test_features.shape)



np.save("train_features.npy",train_features)
np.save("train_labels.npy",train_labels)
np.save("test_features.npy",test_features)
np.save("test_labels.npy",test_labels)
np.save("train_names.npy",train_names)
np.save("test_names.npy",test_names)


