import numpy as np
from classify_gaze_IVT import classify_raw_IVT
import matplotlib.pyplot as plt


exp_file_np = 'example_data.npy'
exp_file_np = 'data/gen_gaze/012_alpha3.5_exp18_softmax2.5.npy'

exp_file_np = 'data/gen_gaze/012_39_observers.npy'

def get_screen_params():
    screen_size_mm = np.asarray([28650, 50930])	 #Lasciare così
    screen_res = np.asarray([720, 1280])		 #Risoluzione del video
    return {'pix_per_mm': screen_res / screen_size_mm,
            'screen_dist_mm': 600,
            'screen_res': screen_res}


gen_scans =np.load(exp_file_np, allow_pickle=True).item()
for k,gen_data in gen_scans.items() :
    #Load raw gaze data
    xy_raw_gaze_data = np.load(exp_file_np, allow_pickle=True).item()[0]
    #import code; code.interact(local=locals())
    gaze_sample_rate = 240.

    screen_params = get_screen_params()
    #Get fixations and their duration from raw gaze data
    gen_fix_plus_dur = classify_raw_IVT(xy_raw_gaze_data, gaze_sample_rate, screen_params)

    print('\nFixations and relative duration in milliseconds:')
    print(len(gen_fix_plus_dur))
    print((gen_fix_plus_dur))

    #import code; code.interact(local=locals())



#Plot!
plt.plot(xy_raw_gaze_data[:,0], xy_raw_gaze_data[:,1], '-x', zorder=-1, label='Raw gaze data')
plt.gca().invert_yaxis()
plt.grid()
plt.scatter(gen_fix_plus_dur[:,0], gen_fix_plus_dur[:,1], c='r', s=100, zorder=1, label='Classified fixations')
plt.legend()
plt.show()
