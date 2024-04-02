import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from GazeParser.ScanMatch import ScanMatch
import multimatch_gaze as mmg
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import multiprocessing as mp
import os
import os.path
import cv2
from RQA import RQA


def generate_random_scans(n_subj, w, h, spatial_dist='gauss'):
	scans = {}

	for s in range(n_subj):
		n_fix = int(np.random.randn()*6 + 45)

		if spatial_dist == 'uniform':
			x_coord = np.random.randint(0, w-1, n_fix)
			y_coord = np.random.randint(0, h-1, n_fix)
		else:
			mu_w = w//2
			mu_h = h//2
			sd_w = w//10
			sd_h = h//10
			x_coord = np.random.normal(mu_w, sd_w, n_fix)
			y_coord = np.random.normal(mu_h, sd_h, n_fix)

		dur = np.random.uniform(1,2000, n_fix)

		scan = np.hstack([x_coord[:,np.newaxis], y_coord[:,np.newaxis], dur[:,np.newaxis]])

		scans[s] = scan

	return scans



experiment = '_random.npy'
expSM = '_random_dur50.npy'

real_dir = 'data/find_dataset/Our_database/fix_and_dur_data/'
out_scores = 'saved/scores/'
video_dir = 'data/find_dataset/Our_database/raw_videos/'

compute_SM = True
compute_MM = True
compute_RQA = True

names = os.listdir(video_dir)

for name in names:

	compute_scores = True

	if not name.endswith('.avi'):
		continue

	vid_name = name.split('.')[0]

	print('\n\tProcessing ' + name + '\n')

	real_scans = sio.loadmat(real_dir + str(int(vid_name)) + '.mat')
	real_scans = real_scans['curr_v_all_s']
	n_real_sbj = real_scans.shape[0]

	gen_class_scan = generate_random_scans(4, 1280, 720)

	if compute_scores:

		matchObject = ScanMatch(Xres=1280, Yres=720, Xbin=14, Ybin=8, TempBin=50)

		print('\nStarting ScanMatching between real and Generated data...')

		gen_scores_sm = []
		gen_scores_mm = []
		gen_RQA_scores = []
		n_gen_sbj = len(gen_class_scan)

		for i in range(n_gen_sbj):
			for j in range(n_real_sbj):

				real_fix = real_scans[j][0].astype(int)
				gen_fix = gen_class_scan[i]

				if compute_RQA:
					rqa = RQA(real_fix, gen_fix)
					rqa_scores = rqa.compute_rqa_metrics()
				if compute_MM:
					scan1_pd = pd.DataFrame({'start_x': real_fix[:,0], 'start_y': real_fix[:,1], 'duration': real_fix[:,2]})
					scan2_pd = pd.DataFrame({'start_x': gen_fix[:,0], 'start_y': gen_fix[:,1], 'duration': gen_fix[:,2]})
				if compute_SM:
					s1 = matchObject.fixationToSequence(real_fix).astype(int)
					s2 = matchObject.fixationToSequence(gen_fix).astype(int)

				if compute_SM:
					(score, _, _) = matchObject.match(s1, s2)
				if compute_MM:
					mm_scores = mmg.docomparison(scan1_pd.to_records(), scan2_pd.to_records(), screensize=[1280, 720])

				if compute_SM:
					print('\nScanMatch Score:')
					print(score)
				if compute_MM:
					print('MultiMatch Score:')
					print(mm_scores)
				if compute_RQA:
					print('RQA Score:')
					print(rqa_scores)

				if compute_SM:
					gen_scores_sm.append(score)
				if compute_MM:
					gen_scores_mm.append(mm_scores)
				if compute_RQA:
					gen_RQA_scores.append(rqa_scores)

		if compute_SM:
			gen_scores_sm = np.array(gen_scores_sm)
		if compute_MM:
			gen_scores_mm = np.vstack(gen_scores_mm)
		if compute_RQA:
			gen_RQA_scores = np.vstack(gen_RQA_scores)
		
		if compute_SM:
			np.save(out_scores + 'gen_scores_SM_' + vid_name + expSM, gen_scores_sm)
		if compute_MM:
			np.save(out_scores + 'gen_scores_MM_' + vid_name + experiment, gen_scores_mm)
		if compute_RQA:
			np.save(out_scores + 'gen_scores_RQA_' + vid_name + experiment, gen_RQA_scores)