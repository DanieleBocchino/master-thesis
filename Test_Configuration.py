# from __future__ import print_function
import shutil
import os
import numpy as np
from ThompsonSampling.Generate_Plot import generate_exponential_Q
from ThompsonSampling.Library_Test import library_test_simulation
from gaze import Gaze
from MVT_gaze_sampler import GazeSampler
from video import Video
from feature_maps import Feature_maps
from sklearn.linear_model import LinearRegression
from utils import normalize, center, compute_density_image, softmax, split_events
import matplotlib.pyplot as plt
from pylab import *
from skimage.draw import circle_perimeter
import yaml
import imageio
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import pandas as pd
from utils import load_config

config =load_config()

# ___________________________ Parameters _______________________________

vidDir = config['TEST CONFIG']['vidDir']
gazeDir = config['TEST CONFIG']['gazeDir']
# gazeDir = 'data/fix_and_dur_data/'
gazeGen = config['TEST CONFIG']['gazeGen']
dynmapDir = config['TEST CONFIG']['dynmapDir']
facemapDir = config['TEST CONFIG']['facemapDir']
outputs_dir = config['TEST CONFIG']['outputs_dir']
d_rate = config['TEST CONFIG']['d_rate']
n_samples = config['TEST CONFIG']['n_samples']
fs = config['TEST CONFIG']['fs']
train_observers = config['TRAIN']['train_observers']
test_observers = config['TEST CONFIG']['observers']
curr_vid_name = config['TRAIN']['curr_vid_name']
phi = config['TEST CONFIG']['phi']
kappa = config['TEST CONFIG']['kappa']
logi= config['TEST CONFIG']['logistic']
full_simulation = config['STATS']['full simulation']

if full_simulation:
    
    #Clear and recreate the result directory
    result_dir = f'RESULT/{curr_vid_name[:-4]}/'
    os.makedirs(result_dir, exist_ok=True)

    config['STATS']['result dir'] = result_dir

    # Save the modified configuration to the file
    with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
    

gazeObj = Gaze(config['TEST CONFIG']['gazeDir'])
videoObj = Video(config['TEST CONFIG']['vidDir'])
featMaps = Feature_maps(config['TEST CONFIG']['dynmapDir'],
                        config['TEST CONFIG']['facemapDir'])

# ___________________________ Functions _______________________________


def get_all_patches(featMaps):
    patches = []
    for fmap in featMaps.all_fmaps:
        for patch in fmap.patches:
            patches.append(patch)
    return patches

# ___________________________ Main _______________________________


scanPath = {}
gen_scan = {}

# DA CAMBIARE DOPO TUNING
if config["STATS"]["launch exp"]:
    experiment = config["TEST CONFIG"]["experiment"]
else:
    experiment = f'_{config["TEST CONFIG"]["observers"]}_obs_phi_{config["TEST CONFIG"]["phi"]}_k_{config["TEST CONFIG"]["kappa"]}_log_{config["TEST CONFIG"]["logistic"]}'
    config['TEST CONFIG']['experiment']  = experiment
    # Save the modified configuration to the file
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)

fname = f'{gazeGen}{curr_vid_name[:-4]}{experiment}.npy'

if os.path.isfile(fname):
    print('\nVideo ' + curr_vid_name + ' already processed... Skipping!\n')

outputs_vid_dir = f'{outputs_dir}{curr_vid_name[:-4]}/'

if os.path.exists(outputs_vid_dir):
    shutil.rmtree(outputs_vid_dir)
    os.mkdir(outputs_vid_dir)


outputs_dir_scanpath = f'{outputs_vid_dir}scanpath/'
os.makedirs(outputs_dir_scanpath, exist_ok=True)

outputs_dir_exponential_Q = f'{outputs_vid_dir}exponential_Q/'
os.makedirs(outputs_dir_exponential_Q + '/gif/', exist_ok=True)

print('\n\n\t\t' + curr_vid_name + '\n')
# Gaze -------------------------------------------------------------------------------------------------------
gazeObj.load_gaze_data(curr_vid_name)

# Video ------------------------------------------------------------------------------------------------------
videoObj.load_video(curr_vid_name)
FOAsize = int(np.max(videoObj.size)/10)

# Feature Maps -----------------------------------------------------------------------------------------------
featMaps.load_feature_maps(
    curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)

# Gaze Sampler -----------------------------------------------------------------------------------------------
gazeSampler = GazeSampler(video_fr=videoObj.frame_rate, phi=phi, kappa=kappa)


nFrames = min([len(videoObj.videoFrames), featMaps.num_sts,
                featMaps.num_speak, featMaps.num_nspeak])
wd = int(videoObj.vidWidth * d_rate / 100)
hd = int(videoObj.vidHeight * d_rate / 100)
tot_dim = wd*hd

total_obs_gif_df = pd.DataFrame()

print(f'Number of observers: {test_observers}')

for observer in tqdm(range(test_observers)):

    filtered_state_means = np.zeros((nFrames, n_samples))
    filtered_state_covariances = np.zeros((nFrames, n_samples, n_samples))
    betas = np.zeros((nFrames, n_samples))
    gazeSampler.sampled_gaze = []

    # For each video frame
    for iframe in tqdm(range(nFrames)):

        # Variables Initialization
        frame = videoObj.videoFrames[iframe]
        SampledPointsCoord = []

        frame = videoObj.videoFrames[iframe]
        featMaps.read_current_maps(
            gazeObj.eyedata, iframe, compute_heatmap=True)

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

        if (len(patches) >0):
            # Gaze Sampling --------------------------------------------------------------------------------------------
            gazeSampler.sample(
                iframe=iframe, patches=patches, FOAsize=FOAsize//12)

        # At the end of the loop
        featMaps.release_fmaps()

    scanPath[observer] = np.concatenate(gazeSampler.sampled_gaze)
    gen_scan[observer] = np.concatenate([gazeSampler.gif_df])
    gazeSampler.gif_df = pd.DataFrame()

np.save(fname, scanPath)
np.save(f'{outputs_dir_exponential_Q}gen_exp_Q{experiment}.npy', gen_scan)

if config['STATS']['launch exp'] :
    result_dir = f'{gazeGen}/gen_exp_Q/'
    os.makedirs(result_dir, exist_ok=True)
    np.save(f'{result_dir}/scanpath.npy', scanPath)
    np.save(f'{result_dir}gen_exp_Q{experiment}.npy', gen_scan)
