

import os
import numpy as np
from tqdm import tqdm
from utils import load_config


config = load_config()

scores_dir = config['FULL SIMULATION']['scores directory']


def getMeanScore(scores_dir, type):

    mm_scores = []
    sm_scores = []
    rqa_scores = []
    
    spec_scores_dir = os.path.join(scores_dir, type)

    for score in tqdm(os.listdir(spec_scores_dir)):
        if score.endswith('.npy') and 'MM' in score:
            score_path = os.path.join(spec_scores_dir, score)
            s = np.load(score_path)
            mm_score = np.mean(s)
            mm_scores.append(mm_score)

        if score.endswith('.npy') and 'SM' in score:
            score_path = os.path.join(spec_scores_dir, score)
            s = np.load(score_path)
            sm_score = np.mean(s)
            sm_scores.append(sm_score)

        if score.endswith('.npy') and 'RQA' in score:
            score_path = os.path.join(spec_scores_dir, score)
            s = np.load(score_path)
            rqa_score = np.mean(s)
            rqa_scores.append(rqa_score)

    return mm_scores, sm_scores, rqa_scores


if '__main__' == __name__:

    """ random_sm_scores, random_mm_scores, random_rqa_scores = [], [], []
    eco_sm_scores, eco_mm_scores, eco_rqa_scores = [], [], []
    ts_sm_scores, ts_mm_scores, ts_rqa_scores = [], [], []

    random_sm_scores, random_mm_scores, random_rqa_scores = getMeanScore(
        os.path.join(scores_dir, 'random'))
    eco_sm_scores, eco_mm_scores, eco_rqa_scores = getMeanScore(
        os.path.join(scores_dir, 'eco'))
    ts_sm_scores, ts_mm_scores, ts_rqa_scores = getMeanScore(
        os.path.join(scores_dir, 'ts')) """
        
    spec_scores_dir = os.path.join(scores_dir, 'ts')
    file_path = spec_scores_dir + "/gen_scores_MM_{video_name}.npy"
    np.save(file_path, array)
    
    np.save(os.path.join(spec_scores_dir, 'mm_scores.npy'), mm_scores)
