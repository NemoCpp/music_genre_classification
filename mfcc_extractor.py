import numpy 
import sidekit
import pickle
import os


extractor = sidekit.FeaturesExtractor(audio_filename_structure="data/{}",
									feature_filename_structure="features/{}.h5",
									sampling_frequency=22050,
									lower_frequency=100,
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
								feature_filename_structure="features/{}.h5",
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


# test feature extraction
Y = {}
utter_list = os.listdir("data/")
count = 0
for j in utter_list:
	b = server.load(j)
	Y[j] = b[0]
	count += 1
	if(numpy.mod(count,5)==0):
		print "count: ", count

pickle.dump(Y,open("mfcc_features.pkl","wb"))

print("feature extraction done")