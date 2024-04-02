#from __future__ import print_function
import cv2
#import os
import numpy as np
from ThompsonSampling.Library_Test import library_test_simulation
from gaze import Gaze
from MVT_gaze_sampler import GazeSampler
from video import Video
from feature_maps import Feature_maps
from sklearn.linear_model import LinearRegression
from utils import normalize, center, compute_density_image, softmax, split_events
import matplotlib.pyplot as plt
from pylab import *
#from drawnow import drawnow
from skimage.draw import circle_perimeter
#from utils import compute_density_image
import os
import imageio
from pykalman import KalmanFilter
from tqdm import tqdm
import pandas as pd
import multimatch_gaze as mmg
import scipy.io as sio
from utils import load_config

### ___________________________ Parameters _______________________________

config =load_config()

#Directories
vidDir = config['TEST CONFIG']['vidDir']
gazeDir = config['TEST CONFIG']['gazeDir']
dynmapDir = config['TEST CONFIG']['dynmapDir']
facemapDir = config['TEST CONFIG']['facemapDir']
outputs_dir = config['TEST CONFIG']['outputs_dir']
d_rate = config['TEST CONFIG']['d_rate']
n_samples = config['TEST CONFIG']['n_samples']
train_observers = config['TRAIN']['train_observers']
test_observers = config['TEST CONFIG']['observers']
start_gif = config['TEST CONFIG']['start_gif']
end_gif	= config['TEST CONFIG']['end_gif']
curr_vid_name	= config['TRAIN']['curr_vid_name']
gazeGen = config['TEST CONFIG']['gazeGen']
experiment = config['TEST CONFIG']['experiment']
result_dir = config['STATS']['result dir']
full_simulation = config['STATS']['full simulation']

gazeObj = Gaze(gazeDir)
videoObj = Video(vidDir)
featMaps = Feature_maps(dynmapDir, facemapDir)
fig = plt.figure(figsize=(16, 10))
images = []
draw_fig = True
display = False
save_GIF = True


### ___________________________ Functions _______________________________
def get_all_patches(featMaps):
	patches = []	
	for fmap in featMaps.all_fmaps:
		for patch in fmap.patches:
			patches.append(patch)
	return patches

def get_fix_from_scan(scan_dict, nFrames):
	generated_eyedata = np.zeros([2, nFrames, len(scan_dict)])
	fixations = np.zeros([2, nFrames])
	for i,k in enumerate(scan_dict.keys()):
		s = scan_dict[k]
		N = s.shape[0] // 10
		frames = np.split(s, N)
		#fixations = np.median(frames, axis=1)
		for j, f in enumerate(frames):
			if j < nFrames:
				med = np.median(f, axis=0)
				fixations[:,j] = np.median(f, axis=0)
		generated_eyedata[:,:,i] = fixations
	return generated_eyedata


### ___________________________ Main _______________________________


fname = f'{gazeGen}{curr_vid_name[:-4]}{experiment}.npy'
scanPath = {}

print('\n\n\t\t' + curr_vid_name + '\n')
#Gaze -------------------------------------------------------------------------------------------------------		
gazeObj.load_gaze_data(curr_vid_name)

#Video ------------------------------------------------------------------------------------------------------	
videoObj.load_video(curr_vid_name)
FOAsize = int(np.max(videoObj.size)/10)

#Feature Maps ----------------------------------------------------------------------------------------------- 
featMaps.load_feature_maps(curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)

#Gaze Sampler -----------------------------------------------------------------------------------------------
gazeSampler = GazeSampler(videoObj.frame_rate, 3.5, 18)  

nFrames = min([len(videoObj.videoFrames), featMaps.num_sts, featMaps.num_speak, featMaps.num_nspeak])
wd = int(videoObj.vidWidth * d_rate / 100)
hd = int(videoObj.vidHeight * d_rate / 100)
tot_dim = wd*hd

initial_state_mean = np.zeros(5)
initial_state_mean[1] = 1
trans_cov = np.eye(5)
vidHeight = int(featMaps.vidHeight * d_rate)
vidWidth = int(featMaps.vidWidth * d_rate)
n_dim_obs = vidHeight * vidWidth

filtered_state_means = np.zeros((nFrames, n_samples))
filtered_state_covariances = np.zeros((nFrames, n_samples, n_samples))
betas = np.zeros((nFrames, n_samples))

#generated_scan = np.load('data/gen_gaze/'+curr_vid_name[:-4]+'.npy', allow_pickle=True).item()

#generated_scan = np.load('data/gen_gaze/'+curr_vid_name[:-4]+'.npy', allow_pickle=True).item()
generated_scan = np.load(fname, allow_pickle=True).item()
#import code; code.interact(local=dict(globals(), **locals()))
generated_eyedata = get_fix_from_scan(generated_scan, nFrames)

if full_simulation:
	os.makedirs(f'{result_dir}gif/', exist_ok=True)

for observer in tqdm(range(train_observers, test_observers)):
	
	#For each video frame
	for iframe in tqdm(range(nFrames)):
		
		#Variables Initialization 
		frame = videoObj.videoFrames[iframe]
		SampledPointsCoord = []
	
		frame = videoObj.videoFrames[iframe]
		featMaps.read_current_maps(gazeObj.eyedata, iframe, compute_heatmap=True)
		
		#Center Bias saliency and proto maps
		featMaps.cb.esSampleProtoParameters()
		featMaps.cb.define_patchesDDM()
		#Speaker saliency and proto maps -------------------------------------------------------------------------
		featMaps.speaker.esSampleProtoParameters()
		featMaps.speaker.define_patchesDDM()

		#Non Speaker saliency and proto maps ---------------------------------------------------------------------		
		featMaps.non_speaker.esSampleProtoParameters()
		featMaps.non_speaker.define_patchesDDM()
					
		#Low Level Saliency saliency and proto maps ---------------------------------------------------------------		
		featMaps.sts.esSampleProtoParameters()
		featMaps.sts.define_patchesDDM()
		patches = get_all_patches(featMaps)
	

		if(len(patches) == 6):
			#Gaze Sampling --------------------------------------------------------------------------------------------
			gazeSampler.sample(iframe=iframe, patches=patches, FOAsize=FOAsize//12)
			
			curr_fix = generated_eyedata[:,iframe,:].T
			gen_saliency = compute_density_image(curr_fix, [videoObj.vidWidth, videoObj.vidHeight])
			
			if draw_fig:
				nRows = 2
				nCols = 3

				fig.clf()

				numfig=1
				plt.subplot(nRows,nCols,numfig)
				plt.imshow(frame)
				plt.title('Original Frame')
				
				numfig+=1
				plt.subplot(nRows,nCols,numfig)
				plt.imshow(featMaps.original_eyeMap)
				plt.title('"Real" Fixation Map')
				
				sp = betas[:iframe,3]
				nsp = betas[:iframe,4]
				cb = betas[:iframe,1]
				sts = betas[:iframe,0]
				uni = betas[:iframe,2]
				npoints = len(sp)
				numfig+=1
				plt.subplot(nRows,nCols,numfig)
				plt.plot(sp, label='speaker')
				plt.plot(nsp, label='Non speaker')
				plt.plot(cb, label='CB')
				plt.plot(sts, label='STS')
				plt.plot(uni, label='Uniform')
				plt.legend()
				plt.grid()
				plt.ylim(0, 0.4)
				plt.title('Value')

				numfig+=1
				finalFOA = gazeSampler.allFOA[-1].astype(int)
				plt.subplot(nRows,nCols,numfig)
				BW = np.zeros(videoObj.size)
				rr,cc = circle_perimeter(finalFOA[1], finalFOA[0], FOAsize)
				rr[rr>=BW.shape[0]] = BW.shape[0]-1
				cc[cc>=BW.shape[1]] = BW.shape[1]-1
				BW[rr,cc] = 1
				FOAimg = cv2.bitwise_and(cv2.convertScaleAbs(frame),cv2.convertScaleAbs(frame),mask=cv2.convertScaleAbs(BW))
				plt.imshow(FOAimg)
				plt.title('Focus Of Attention (FOA)')

				#Heat Map
				numfig+=1
				plt.subplot(nRows,nCols,numfig)
				plt.imshow(gen_saliency)
				plt.title('Generated Fixation Map')

				#Scan Path
				numfig+=1
				plt.subplot(nRows,nCols,numfig)
				plt.imshow(frame)
				sampled = np.concatenate(gazeSampler.sampled_gaze)
				plt.plot(sampled[:,0], sampled[:,1], '-x')
				plt.title('Generated Gaze data')

				fig.canvas.draw()
				image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
				image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				if iframe >= start_gif:
					if save_GIF:
						images.append(image)
				if iframe > end_gif:
					break

				if display:
					plt.pause(1/25.)

		#At the end of the loop
		featMaps.release_fmaps()


	#kwargs_write = {'fps':10.0, 'quantizer':'nq'}
	if save_GIF:
		imageio.mimsave(f'{outputs_dir}simulation_' + curr_vid_name[:-4] + '_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+ '.gif', images, fps=10)
		
	if full_simulation:
		imageio.mimsave(f'{result_dir}gif/obs_{observer}'+'.gif', images, fps=10)
