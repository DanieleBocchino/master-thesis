import shutil
import subprocess
import yaml
import os
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import mutual_info_score
import os
import platform
from utils import load_config
from tqdm import tqdm
import shutil
## ___ PARAMS ___ ##

train_script = "Train_Simulation.py"
test_configuration_script = "Test_Configuration.py"
compute_stats_script = "compute_stats.py"
test_script = "Test_Simulation.py"
config = load_config()

## ___ FUNCTIONS ___ ##

def train_function(video):

    if not os.path.exists(f'ThompsonSampling/models/{video}/logistic_ts_model.pkl'):
        subprocess.run(["python", train_script],
                       shell=platform.system() == 'Windows', check=True)
    print(f' \n\n\n END TRAINING : {video} \n\n\n')


def test_function(video):
    if not os.path.exists(f'RESULT/{video}/full_fix_duration.png'):
        subprocess.run(["python", test_configuration_script],
                       shell=platform.system() == 'Windows', check=True)
        subprocess.run(["python", compute_stats_script],
                       shell=platform.system() == 'Windows', check=True)
    print(f' \n\n\n END CONFIGURATION : {video} \n\n\n')
    if not os.path.exists(f'RESULT/{video}/gif/'):
        subprocess.run(["python", test_script],
                       shell=platform.system() == 'Windows', check=True)
    print(f' \n\n\n END SIMULATION : {video} \n\n\n')
## ___ MAIN ___ ##


def add_to_exclude_list(video):
    with open('exclude_list.txt', 'a') as f:
        f.write(video)
        f.write('\n')


def copy_files(source_folder, destination_folder, file_extensions):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file in os.listdir(source_folder):
        if any(file.endswith(ext) for ext in file_extensions):
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy2(source_path, destination_path)
            print(f"Copied {file} to {destination_folder}")

file_extensions = ['.npy']

def check_exclude_list(video):
    with open('exclude_list.txt', 'r') as f:
        return video in f.read()

if __name__ == "__main__":

    config = load_config()
    
    config['STATS']['launch exp'] = False
    config['TEST CONFIG']['Bayesian Optimization'] = False
    video_dir = config['FULL SIMULATION']['video directory']

    for video in tqdm(os.listdir(video_dir)):

        if check_exclude_list(video):
            continue

        elif video.endswith(".mp4"):
            print(f' \n\n\n START VIDEO : {video} \n\n\n')

            config['TRAIN']['curr_vid_name'] = video
            
            # Set path for current video directory in models directory
            config['TRAIN']['models_dir'] = f'ThompsonSampling/models/{video[:-4]}/'

            # Save the modified configuration to the file
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)

            if check_exclude_list(video) == False:
                train_function(video[:-4])

            if check_exclude_list(video) == False:
                test_function(video[:-4])
                
                if os.path.exists(f'outputs/test/{video[:-4]}/scores'):
                    source_folder = f'outputs/test/{video[:-4]}/scores'
                    destination_folder = f'RESULT/{video[:-4]}/scores'
                    os.makedirs(destination_folder, exist_ok=True)
                    copy_files(source_folder, destination_folder, file_extensions)
                    """ else:
                    subprocess.run(["python", compute_stats_script], shell=platform.system() == 'Windows', check=True)
                    source_folder = f'outputs/test/{video[:-4]}/scores'
                    
                    destination_folder = f'RESULT/{video[:-4]}/scores'
                    os.makedirs(destination_folder, exist_ok=True)
                

                    copy_files(source_folder, destination_folder, file_extensions) """
