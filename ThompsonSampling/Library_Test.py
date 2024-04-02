import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from utils import load_config

config =load_config()



def library_test_simulation(X,k = 25):  
    
    ## The base algorithm is embedded in different metaheuristics
    curr_vid_name = config['TRAIN']['curr_vid_name'][:-4]
    #import code; code.interact(local=locals())
    epsilon_greedy=cloudpickle.load(open(f"ThompsonSampling/models/{curr_vid_name}/epsilon_greedy_model.pkl", "rb"))
    adaptive_greedy_perc=cloudpickle.load(open(f"ThompsonSampling/models/{curr_vid_name}/adaptive_greedy_model.pkl", "rb"))
    logistic_ts=cloudpickle.load(open(f"ThompsonSampling/models/{curr_vid_name}/logistic_ts_model.pkl", "rb"))  
    
    models = [epsilon_greedy, adaptive_greedy_perc, logistic_ts]
    #import code; code.interact(local=dict(globals(), **locals()))

    if X.shape[0] == 1:
        X = X.reshape(1,-1)
    #decision function for each model
    dec_fun_egr, dec_fun_agr2, dec_fun_lts = [models[i].decision_function(X)  for i in range(len(models))]
    #import code; code.interact(local=locals())
    
    #create_debug_bar_post(pre_dec_fun_lts[0] , dec_fun_lts[0])

    return dec_fun_egr[0], dec_fun_agr2[0], dec_fun_lts[0] 

def create_debug_bar_post(data_pre, data_post):
            
    if len(data_pre) == 6 and len(data_post) == 6:
                
        # Create x-axis indices
        indices = np.arange(len(data_pre))

        # Set the width of each bar
        bar_width = 0.35

        # Create the figure and axes
        fig, ax = plt.subplots()

        # Create the first bar plot
        ax.bar(indices, data_pre, bar_width, label='Data pre-softmax', )

        # Shift the indices for the second bar plot
        indices_shifted = indices + bar_width

        # Create the second bar plot
        ax.bar(indices_shifted, data_post, bar_width, label='Data post-softmax')

        # Set the labels, title, and legend
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Side-by-Side Bar Plot')
        ax.legend()

        # Adjust the layout to avoid overlapping of bars
        plt.tight_layout()

        # Show the plot
        plt.show()
