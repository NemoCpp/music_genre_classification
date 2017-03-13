import sidekit
import pickle
import sidekit
import os
import sys
import multiprocessing
import matplotlib.pyplot as mpl
import logging
import numpy as np

utter_list = np.array(os.listdir("chunks_data/all_train/"))

print "data loaded", utter_list.shape



# logging.basicConfig(filename='log/rsr2015_ubm-gmm.log',level=logging.DEBUG)
distribNb = 64  # number of Gaussian distributions for each GMM
audioDir = 'chunks_data/all_train/'

# Automatically set the number of parallel process to run.
# The number of threads to run is set equal to the number of cores available
# on the machine minus one or to 1 if the machine has a single core.
nbThread = max(multiprocessing.cpu_count()-1, 1)

extractor = sidekit.FeaturesExtractor(audio_filename_structure="chunks_data/all_train/{}",
									feature_filename_structure="chunks_features/all_train/{}.h5",
									sampling_frequency=None,
									lower_frequency=200,
									higher_frequency=4000,
									filter_bank="log",
									filter_bank_size=26,
									window_size=0.025,
									shift=0.01,
									ceps_number=13,
									# vad="snr",
									# snr=10,
									pre_emphasis=0.97,
									save_param=["energy", "cep", "fb"],
									keep_all_features=True)

server = sidekit.FeaturesServer(features_extractor=extractor,
								feature_filename_structure="chunks_features/all_train/{}.h5",
								# sources=None,
								dataset_list=["energy", "cep", "fb"],
								mask="[0-12]",
								feat_norm="cmvn",
								global_cmvn=None,
								dct_pca=False,
								dct_pca_config=None,
								sdc=False,
								# sdc_config=(1,3,7),
								delta=True,
								double_delta=True,
								delta_filter=None,
								context=None,
								traps_dct_nb=None,
								rasta=False,
								keep_all_features=True)


print('Train the UBM by EM')
# Extract all features and train a GMM without writing to disk
ubm = sidekit.Mixture()
llk = ubm.EM_split(server, utter_list, distribNb)#, num_thread=nbThread)
pickle.dump(ubm,open("ubm_64.pkl","wb"))
ubm.write('gmm/ubm_train_64.h5')
