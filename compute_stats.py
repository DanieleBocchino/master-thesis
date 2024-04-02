import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from GazeParser.ScanMatch import ScanMatch
import multimatch_gaze as mmg
import pandas as pd
import multiprocessing as mp
import os
import os.path
import cv2
from RQA import RQA
from tqdm import tqdm
from IVT.classify_gaze_IVT import classify_raw_IVT
from ThompsonSampling.Generate_Plot import create_direction_saccade, create_distribution_duration, create_wide_saccade, generate_exponential_Q, generate_kde_plot
from utils import load_config

config =load_config()

# ___ PARAMETERS ___#

gaze_sample_rate = 240.
fs = 250.
screen_res_x = 1024
screen_res_y = 768
screen_size_x = 0.4
screen_size_y = 0.3
dist_from_screen = 0.57

np.set_printoptions(suppress=True)

display = False
save_imgs = False
compute_SM = True
compute_MM = True
compute_RQA = True
classify = True
compute_scores = True
duration_distribution = True
exponential_Q = True

# experiment = '_39_obs.npy'
# experiment = '_obs_phi_3.5_kappa_1.npy'

outputs_dir = 'outputs/test/'

generated_dir = 'data/gen_gaze/'
real_dir = 'data/find_Dataset/Our_database/fix_and_dur_data/'

video_dir = '/data/find_Dataset/Our_database/raw_videos/'

names = os.listdir(generated_dir)


# ___________________________ Parameters _______________________________

gaze_sample_rate = config['STATS']['gaze_sample_rate']
fs = config['STATS']['fs']
screen_res_x = config['STATS']['screen_res_x']
screen_res_y = config['STATS']['screen_res_y']
screen_size_x = config['STATS']['screen_size_x']
screen_size_y = config['STATS']['screen_size_y']
dist_from_screen = config['STATS']['dist_from_screen']
display = config['STATS']['display']
save_imgs = config['STATS']['save_imgs']
compute_SM = config['STATS']['compute_SM']
compute_MM = config['STATS']['compute_MM']
compute_RQA = config['STATS']['compute_RQA']
classify = config['STATS']['classify']
compute_scores = config['STATS']['compute_scores']
duration_distribution = config['STATS']['duration_distribution']
exponential_Q = config['STATS']['exponential_Q']
outputs_dir = config['STATS']['outputs_dir']
generated_dir = config['STATS']['generated_dir']
real_dir = config['STATS']['real_dir']
video_dir = config['STATS']['video_dir']
saccade_analysis = config['STATS']['saccade_analysis']
video_name = config['TRAIN']['curr_vid_name']
scores_dir = config['FULL SIMULATION']['scores directory']

np.set_printoptions(suppress=True)
names = os.listdir(generated_dir)

vid_name = video_name[:-4]
cam = cv2.VideoCapture(video_dir + vid_name + '.mp4')
fps = int(np.round(cam.get(cv2.CAP_PROP_FPS)))
fs = fps * 10.
# ___________________________ Functions _______________________________

def get_screen_params():
    screen_size_mm = np.asarray([28650, 50930])  # Lasciare così
    screen_res = np.asarray([720, 1280])  # Risoluzione del video
    return {'pix_per_mm': screen_res / screen_size_mm,
            'screen_dist_mm': 600,
            'screen_res': screen_res}


def make_score(n_gen_sbj, n_real_sbj, gen_class_scan, real_scans, type_score):

    if compute_SM:
        matchObject = ScanMatch(Xres=1280, Yres=720,
                                Xbin=14, Ybin=8, TempBin=50)
        gen_scores_sm = []
    if compute_MM:
        gen_scores_mm = []
    if compute_RQA:
        gen_RQA_scores = []

    # import code; code.interact(local=dict(globals(), **locals()))

    def compute_scores(n_first, n_last, first_scans, last_scans, type_score):

        for i in tqdm(range(n_first)):
            for j in range(i+1, n_last):
                if type_score == 'REAL_VS_REAL':
                    real_fix = first_scans[j][0].astype(int)
                    gen_fix = last_scans[i][0].astype(int)

                elif type_score == 'GEN_VS_GEN':
                    real_fix = first_scans[j]
                    gen_fix = last_scans[i]

                else:
                    real_fix = first_scans[j][0].astype(int)
                    gen_fix = last_scans[i]

                if compute_RQA:
                    rqa = RQA(real_fix, gen_fix)
                    rqa_scores = rqa.compute_rqa_metrics()
                if compute_MM:
                    scan1_pd = pd.DataFrame(
                        {'start_x': real_fix[:, 0], 'start_y': real_fix[:, 1], 'duration': real_fix[:, 2]})
                    scan2_pd = pd.DataFrame(
                        {'start_x': gen_fix[:, 0], 'start_y': gen_fix[:, 1], 'duration': gen_fix[:, 2]})
                if compute_SM:
                    s1 = matchObject.fixationToSequence(real_fix).astype(int)
                    s2 = matchObject.fixationToSequence(gen_fix).astype(int)

                if compute_SM:
                    (score, _, _) = matchObject.match(s1, s2)
                if compute_MM:
                    mm_scores = mmg.docomparison(scan1_pd.to_records(
                    ), scan2_pd.to_records(), screensize=[1280, 720])

                if compute_SM:
                    gen_scores_sm.append(score)
                if compute_MM:
                    gen_scores_mm.append(mm_scores)
                if compute_RQA:
                    gen_RQA_scores.append(rqa_scores)

    print(f'\t {type_score} \n')

    if type_score == 'REAL_VS_REAL':
        compute_scores(n_first=n_real_sbj, n_last=n_real_sbj,
                       first_scans=real_scans, last_scans=real_scans, type_score=type_score)

    elif type_score == 'GEN_VS_GEN':
        compute_scores(n_first=n_gen_sbj, n_last=n_gen_sbj, first_scans=gen_class_scan,
                       last_scans=gen_class_scan, type_score=type_score)

    else:
        compute_scores(n_first=n_real_sbj, n_last=n_gen_sbj, first_scans=real_scans,
                       last_scans=gen_class_scan, type_score=type_score)

    if compute_SM:
        gen_scores_sm = np.array(gen_scores_sm)
    if compute_MM:
        gen_scores_mm = np.vstack(gen_scores_mm)
    if compute_RQA:
        gen_RQA_scores = np.vstack(gen_RQA_scores)

    if compute_SM:
        np.save(out_scores + '/gen_scores_SM_' + type_score +
                '_' + vid_name + experiment, gen_scores_sm)
    if compute_MM:
        np.save(out_scores + '/gen_scores_MM_' + type_score +
                '_' + vid_name + experiment, gen_scores_mm)
    if compute_RQA:
        np.save(out_scores + '/gen_scores_RQA_' + type_score +
                '_' + vid_name + experiment, gen_RQA_scores)

    return gen_scores_sm, gen_scores_mm, gen_RQA_scores




# ____NEW_MAIN _____

experiment = f'{config["TEST CONFIG"]["experiment"]}.npy'

### CREATE DIRECTORIES ###

outputs_vid_dir = f'{outputs_dir}{vid_name}/'
os.makedirs(outputs_vid_dir, exist_ok=True)

out_classified = f'{outputs_vid_dir}classified/'
out_scores = f'{outputs_vid_dir}scores/'
out_imgs = f'{outputs_vid_dir}plot/'

os.makedirs(out_classified, exist_ok=True)
os.makedirs(out_scores, exist_ok=True)
os.makedirs(out_imgs, exist_ok=True)

# and os.path.isfile(out_scores + '/real_scores_SM_' + vid_name + experiment):
if os.path.isfile(out_classified + '/gen_gaze_classified_' + vid_name + experiment):
    classify = False
# continue
""" if os.path.isfile(out_scores + '/gen_scores_SM_' + vid_name + expSM):
    compute_scores = False
    #continue """

# import code; code.InteractiveConsole(locals=globals()); code.interact(local=locals())

real_scans = sio.loadmat(real_dir + vid_name.lstrip('0') + '.mat')
real_scans = real_scans['curr_v_all_s']
n_real_sbj = real_scans.shape[0]

# dictionary, 0..9 keys (one for each fake sbj)
gen_scans = np.load(generated_dir + vid_name +
                    experiment, allow_pickle=True).item()
gen_class_scan = {}

# gen_gaze_sampler = [ gp, curr_fix_dur, curr_rho, A, Q, dist, jump_prob]
if config['STATS']['launch exp']:
    gen_gaze_sampler = (np.load(
        f'{generated_dir}/gen_exp_Q//gen_exp_Q{experiment}', allow_pickle=True)).flat[0]
else:
    gen_gaze_sampler = (np.load(
        f'{outputs_vid_dir}exponential_Q/gen_exp_Q{experiment}', allow_pickle=True)).flat[0]


if duration_distribution:

    # this function is used to get the duration of each fixation
    def get_dur_for_gen(gen_gaze_sampler):
        dur = []
        for i in range(len(gen_gaze_sampler)):
            res = []
            sum = 0.0
            for j in range(len(gen_gaze_sampler[i])):
                if j == 0 or round(gen_gaze_sampler[i][j][2], 2) != round(gen_gaze_sampler[i][j - 1][2], 2):
                    res.append(sum)
                    sum = 0.0
                # here convert the duration in ms because the original is in seconds
                sum += int(round(gen_gaze_sampler[i][j][1] * 1000))
            dur.append(np.array(res))
        return dur

    create_distribution_duration(real_scans=real_scans, gen_dur=get_dur_for_gen(
        gen_gaze_sampler), n_real=n_real_sbj,  experiment=experiment, outputs_dir=out_imgs)

if exponential_Q:
    generate_exponential_Q(gen_gaze_sampler, out_imgs, vid_name)

if classify:
    screen_params = get_screen_params()
# custom classify
    xy_raw_gaze_data = gen_scans[0]

# Generated Data classification
    n_subj = len(gen_scans)
    results = [classify_raw_IVT(
        gen_data, gaze_sample_rate, screen_params) for k, gen_data in gen_scans.items()]

    for i, sp in enumerate(results):
        gen_class_scan[i] = sp

    np.save(out_classified + 'gen_gaze_classified_' +
            vid_name + experiment, gen_class_scan)
    print('\nClassification concluded...')
else:

    print('\nLoading Classified Generated data...')
    gen_class_scan = np.load(
        out_classified + 'gen_gaze_classified_' + vid_name + experiment, allow_pickle=True).item()

if compute_scores:

    n_gen_sbj = len(gen_class_scan)

    type_score = ['REAL_VS_REAL', 'GEN_VS_REAL', 'GEN_VS_GEN']

    results = []

    for t in type_score:
        gen_scores_sm, gen_scores_mm, gen_RQA_scores = make_score(
            n_gen_sbj=n_gen_sbj, n_real_sbj=n_real_sbj, gen_class_scan=gen_class_scan, real_scans=real_scans, type_score=t)

        results.append({'sm': gen_scores_sm, 'mm': gen_scores_mm, 'rqa': gen_RQA_scores})
        
        if t == 'GEN_VS_REAL':
            spec_scores_dir = os.path.join(scores_dir, 'ts')
            np.save(f"{spec_scores_dir}/gen_scores_MM_{vid_name}.npy", gen_scores_mm)
            np.save(f"{spec_scores_dir}/gen_scores_SM_{vid_name}.npy", gen_scores_sm)
            np.save(f"{spec_scores_dir}/gen_scores_RQA_{vid_name}.npy", gen_RQA_scores)

    np.save(out_scores + 'gen_scores_' + vid_name + experiment, results)
    generate_kde_plot(results=results, vid_name=vid_name,
                        experiment=experiment, outputs_dir=out_imgs)
    
if saccade_analysis:
            
    real_dirs = []
    real_amps = []
    gen_dirs = []
    gen_amps = []
    
    #Generated
    for k in gen_class_scan.keys():
        curr_scan = gen_class_scan[k][:,0:2]
        nfix = curr_scan.shape[0]
        for f in range(nfix-1):
            curr_fix2 = curr_scan[f+1,:]
            curr_fix1 = curr_scan[f,:]
            direction = np.arctan2(curr_fix2[1]-curr_fix1[1], curr_fix2[0]-curr_fix1[0]) + np.pi
            amplitude = np.linalg.norm(curr_fix2-curr_fix1)
            gen_dirs.append(direction)
            gen_amps.append(amplitude)
    #Real
    for k in range(n_real_sbj):
        curr_scan = real_scans[k][0][:,0:2].astype(int)
        nfix = curr_scan.shape[0]
        for f in range(nfix-1):
            curr_fix2 = curr_scan[f+1,:]
            curr_fix1 = curr_scan[f,:]
            direction = np.arctan2(curr_fix2[1]-curr_fix1[1], curr_fix2[0]-curr_fix1[0]) + np.pi
            amplitude = np.linalg.norm(curr_fix2-curr_fix1)
            real_dirs.append(direction)
            real_amps.append(amplitude)
    real_amps = np.array(real_amps)
    real_dirs = np.array(real_dirs)
    gen_amps = np.array(gen_amps)
    gen_dirs = np.array(gen_dirs)
    
    create_wide_saccade( real_amps =real_amps, gen_amps=gen_amps, n_real = n_real_sbj,   experiment=experiment, outputs_dir=out_imgs)
    create_direction_saccade( real_dirs =real_dirs, gen_dirs=gen_dirs,  experiment=experiment, outputs_dir=out_imgs)
    print('Saccade analysis concluded...')