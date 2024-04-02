import os
import numpy as np
from ThompsonSampling.BOF import BoF
from ThompsonSampling.CMAB import ContextualMAB
from ThompsonSampling.Generate_Plot import comparison_ts_standScaler, generate_expected_reward
from ThompsonSampling.Library_Train import library_train_simulation
from ThompsonSampling.Offline_Analysis import offline_analysis
from ThompsonSampling.ThompsonSampler import TS
import pandas as pd
from gaze import Gaze
from video import Video
from patch_race_DDM import race_DDM
from feature_maps import Feature_maps
import matplotlib.pyplot as plt
from pylab import *
from skimage.draw import circle_perimeter
import os.path
import scipy.stats as stats
import scipy.io as sio
from tqdm import tqdm
import shutil
from sklearn.kernel_approximation import RBFSampler
from scipy.cluster.vq import kmeans, vq
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pickle
from utils import load_config
import yaml

config =load_config()

#___________________________ Parameters _______________________________

vidDir = config['TRAIN']['vidDir']
gazeDir = config['TRAIN']['gazeDir']
dynmapDir = config['TRAIN']['dynmapDir']
facemapDir = config['TRAIN']['facemapDir']
racers = config['TRAIN']['racers']
curr_vid_name = config['TRAIN']['curr_vid_name']
outputs_dir = config['TRAIN']['outputs_dir']
save_GIF = config['TRAIN']['save_GIF']

n_simulation = config['TRAIN']['n_simulation']
prev_data = config['TRAIN']['prev_data']
num_observer = config['TRAIN']['num_observer']
n_dim = config['TRAIN']['n_dim']
lambda_ = config['TRAIN']['lambda_']
alpha = config['TRAIN']['alpha']  

# Set path for current video directory in models directory
config['TRAIN']['models_dir'] = f'ThompsonSampling/models/{curr_vid_name[:-4]}/' 
# Convert the YAML data to a string
config_str = yaml.dump(config['TRAIN'])

# Save the modified configuration to the file
with open('config.yaml', 'w') as f:
        yaml.dump(config, f)

models_dir = config['TRAIN']['models_dir']       

# ___________________________ Parameters _______________________________
gazeObj = Gaze(gazeDir)
videoObj = Video(vidDir)
featMaps = Feature_maps(dynmapDir, facemapDir)
scanPath = {}


# ___________________________ Functions _______________________________


def get_fix_from_scan(scan_dict, nFrames):
    generated_eyedata = np.zeros([2, nFrames, len(scan_dict)])
    fixations = np.zeros([2, nFrames])
    for i, k in enumerate(scan_dict.keys()):
        s = scan_dict[k]
        N = s.shape[0] // 10
        frames = np.split(s, N)
        for j, f in enumerate(frames):
            if j < nFrames:
                med = np.median(f, axis=0)
                fixations[:, j] = np.median(f, axis=0)
        generated_eyedata[:, :, i] = fixations
    return generated_eyedata


def get_all_patches(featMaps):
    patches = []
    for fmap in featMaps.all_fmaps:
        for patch in fmap.patches:
            patches.append(patch)
    return patches


def sample_values(prior_values, s):
    values = []
    if s == 0:
        return prior_values
    else:
        for v in prior_values:
            # sample rectified normal
            values.append(np.max([0.01, stats.norm.rvs(v, s)]))
        return np.array(values)

# ___________________________ Main _______________________________


# Gaze -------------------------------------------------------------------------------------------------------
gazeObj.load_gaze_data(curr_vid_name)

# Video ------------------------------------------------------------------------------------------------------
videoObj.load_video(curr_vid_name)
FOAsize = int(np.max(videoObj.size)/10)
diag_size = np.sqrt(videoObj.vidHeight**2 + videoObj.vidWidth**2)
fps = videoObj.frame_rate

# Feature Maps -----------------------------------------------------------------------------------------------
featMaps.load_feature_maps(
    curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)
    
nRows = 2
nCols = 3
fig = plt.figure(figsize=(18, 10))

compute_heatmap = True

# Load generated scanpaths
#generated_scan = np.load('data/gen_gaze/'+curr_vid_name[:-4]+'.npy', allow_pickle=True).item()
#generated_eyedata = get_fix_from_scan(generated_scan, nFrames)

prev_patch = None

# Thompson Sampling -----------------------------------------------------------------------------------------------

# _____________ TS options __________________


mat_file = sio.loadmat(gazeDir + curr_vid_name[:-4] + '.mat')
mat_data = np.vstack(mat_file['curr_v_all_s'][num_observer])
round_df = pd.DataFrame({'k': [], 'x': [], 'reward': []})  # Ts  analysis df
analysis_df = pd.DataFrame({'x': [], 'reward': []})  # offline analysis df
experiment_df = pd.DataFrame()  # temp df
offline_analysis_df = pd.DataFrame()  # temp df
offline_contextual_library_df = pd.DataFrame()  # temp df

nFrames = min([len(videoObj.videoFrames), featMaps.num_sts,
              featMaps.num_speak, featMaps.num_nspeak, len(mat_data)])

# lambda e alpha bilanciano exploration e exploitation
ts = TS(lambda_=lambda_, alpha=alpha, n_dim=n_dim)

# _____________ CREATE Directories ________________

# ignore output
shutil.rmtree('outputs') if os.path.exists(f'{outputs_dir}csv/') else None
os.makedirs(f'{outputs_dir}csv/',  exist_ok=True)
os.makedirs(f'{outputs_dir}plot/', exist_ok=True)

shutil.rmtree(models_dir) if os.path.exists(models_dir) else None
os.makedirs(models_dir,  exist_ok=True)

# _____________ FeatMaps Creation __________________

iFrame_patches = []

for iframe in tqdm(range(nFrames)):  # hjhkkh

    frame = videoObj.videoFrames[iframe]
    featMaps.read_current_maps(
        gazeObj.eyedata, iframe, compute_heatmap=compute_heatmap)

    # Center Bias saliency and proto maps
    featMaps.cb.esSampleProtoParameters()
    featMaps.cb.define_patchesDDM()
    # Speaker saliency and proto maps -------------------------------------------------------------------------
    featMaps.speaker.esSampleProtoParameters()
    featMaps.speaker.define_patchesDDM()

    # Non Speaker saliency and proto maps ---------------------------------------------------------------------
    featMaps.non_speaker.esSampleProtoParameters()
    featMaps.non_speaker.define_patchesDDM()

    # Low Level Saliency saliency and proto maps ---------------------------------------------------------------
    featMaps.sts.esSampleProtoParameters()
    featMaps.sts.define_patchesDDM()
    patches = get_all_patches(featMaps)
    if (len(patches) > 0):
        iFrame_patches.append(patches)

    featMaps.release_fmaps()


# _____________ TS Simulation __________________
#for observer in tqdm(range(len(mat_file['curr_v_all_s']))):
for observer in tqdm(range(config['TRAIN']['train_observers'])):

    for iframe in range(nFrames):
        if iframe == 0:

            c_mab = ContextualMAB(
                patches=iFrame_patches[iframe], mat_file=mat_data)
            ts.curr_patch_ts = iFrame_patches[iframe][iframe].center
            ts.n_bandits = len(iFrame_patches[iframe])
            ts.buffer_size = nFrames

        if not np.array_equal(prev_data, mat_file['curr_v_all_s'][observer][0][iframe]):

            prev_data = mat_file['curr_v_all_s'][observer][0][iframe]
            ts.TS_learning(
                patches=iFrame_patches[iframe],  iframe=iframe, c_mab=c_mab, analysis_df=analysis_df)

            round_df = ts.round_df
            analysis_df = ts.analysis_df
            contextual_lib_df = ts.contextual_library_df

        # import code; import code; code.interact(local=locals())

        # adding information about simulation and decision policy in the df
    round_df = round_df.assign(simul_id=observer)

    # accumulating in experiment df
    experiment_df = pd.concat([experiment_df, round_df])
    
    #non necessaria
    #offline_analysis_df = pd.concat([offline_analysis_df, analysis_df])
    
    offline_contextual_library_df = pd.concat(
        [offline_contextual_library_df, contextual_lib_df])

experiment_df.to_csv(f'{outputs_dir}csv/out.csv')
offline_contextual_library_df.to_csv(f'{outputs_dir}csv/output_CB_library.csv')


# _____________ TS LIBRARY Simulation __________________

k_bandit = experiment_df.k.values.astype(int)

try:
    def res_lambda(lst): return np.array([lst[i] for i in range(len(lst))])

    #Train test split
    """ test_size = int(X.shape[0] * 0.3)
    X_train = X[:test_size,:]
    X_test = X[test_size:,:]
    y_train = yn[:test_size]
    y_test = yn[test_size:] """
    #import code; code.interact(local=locals())
    X_train = res_lambda(offline_contextual_library_df.X.values)[:-1,:]
    y_train = res_lambda(offline_contextual_library_df.y.values)[1:]

    X_train_bof, codebook = BoF(X_train, codebook=None, k=25, normalize=True)	#train the BoF model

    np.save(f'{models_dir}codebook.npy', codebook)

    rbf_features = RBFSampler(gamma=0.01, random_state=1, n_components=400)
    X_features = rbf_features.fit_transform(X_train_bof)


    # Save the sampler object to a file
    filename = f'{models_dir}rbf_features.pkl'
    with open(filename, "wb") as file:
        pickle.dump(rbf_features, file)
    
        
    #shift y and x di 1
    library_train_simulation(X=X_features,
                            y=y_train,
                            k=k_bandit)

    # Offline Analysis -------------------------------------------------------------------------------------------------------

    #comparison_ts_stdScaler_df = offline_analysis( offline_analysis_df, experiment_df, n_dim)

    # Plot Generation -------------------------------------------------------------------------------------------------------
    generate_expected_reward(experiment_df)
    #comparison_ts_standScaler(comparison_ts_stdScaler_df, n_dim=n_dim)
    
except Exception as e:
    
    with open('exclude_list.txt', 'a') as f:
        f.write(config['TRAIN']['curr_vid_name'])
        f.write('\n')
