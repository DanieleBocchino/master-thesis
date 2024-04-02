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


def black_box_function(observers, phi, kappa, logistic):
       
        config['TEST CONFIG']['observers'] = int(observers.item())
        config['TEST CONFIG']['phi'] = phi.item()
        config['TEST CONFIG']['kappa'] = kappa.item()
        config['TEST CONFIG']['logistic'] = logistic.item()
        config['TEST CONFIG']['experiment']  = f'_{config["TEST CONFIG"]["observers"]}_obs_phi_{config["TEST CONFIG"]["phi"]}_k_{config["TEST CONFIG"]["kappa"]}_log_{config["TEST CONFIG"]["logistic"]}'
        experiment = config['TEST CONFIG']['experiment']
        
        #Clear and recreate the result directory
        custom_result_dir = f'{result_dir}exp{experiment}/'
        os.makedirs(custom_result_dir, exist_ok=True)

        config['STATS']['result dir'] = custom_result_dir
        
        # Convert the YAML data to a string
        config_str = yaml.dump(config['TEST CONFIG'])

        # Save the modified configuration to the file
        with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
        
        # Write the string to a file
        with open(f"{custom_result_dir}config.txt", "w") as f:
                f.write(config_str) 
        
        
        subprocess.run(["python", test_configuration_script], shell=platform.system() == 'Windows' , check=True)
        subprocess.run(["python", compute_stats_script], shell=platform.system() == 'Windows', check=True)
        
        gen  = np.load(f'{custom_result_dir}gen.npy')
        real = np.load(f'{custom_result_dir}real.npy')
        
                
        def uneven_kl_divergence(pk,qk):
                if len(pk)>len(qk):
                        pk = np.random.choice(pk,len(qk))
                elif len(qk)>len(pk):
                        qk = np.random.choice(qk,len(pk))
                e = 1e-10
                return np.sum(pk * np.log(e+pk/(qk+e)))
        
        score =  uneven_kl_divergence(gen,real)

        
        os.remove(f'{custom_result_dir}gen.npy')
        os.remove(f'{custom_result_dir}real.npy') 
        
        return -score


if __name__ == "__main__":
        # Load the configuration file
        config =load_config()
        config['STATS']['launch exp'] = True
        config['TEST CONFIG']['Bayesian Optimization'] = True
        
        if config['TEST CONFIG']['Bayesian Optimization']:
                config['TEST CONFIG']['observers'] = 10
                
        #Clear and recreate the result directory
        result_dir = f'RESULT/'
        # Create the folder if it doesn't exist
        if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
        #shutil.rmtree(result_dir)

        test_configuration_script = "Test_Configuration.py"
        compute_stats_script = "compute_stats.py"

        """  exp_data = [{
        'observers': 10,
        'phi': 5.0 , 
        'kappa': 2.0,
        'logistic' : 28.693611033785626
        }]
        """
        # Get the list of all files and directories in the specified directory
        elements = os.listdir(result_dir)  
        
        # Bounded region of parameter space
        pbounds = { 'observers': (config['TEST CONFIG']['observers'],config['TEST CONFIG']['observers']),'phi': (2, 5), 'kappa': (0, 2), 'logistic': (15, 30)}

        optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=1,
        ) 

        optimizer.maximize(init_points = 5, n_iter = 20 )
                
        print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
        # Write the string to a file
        with open(f"best_config.txt", "w") as f:
                f.write("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"])) 
        f.close()
   
                