from sklearn.linear_model import LogisticRegression
from contextualbandits.online import EpsilonGreedy, AdaptiveGreedy, LogisticTS
import cloudpickle
from copy import deepcopy
from pylab import rcParams
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ThompsonSampling.Generate_Plot import generate_plot, generate_hist
from utils import softmax
import yaml
from utils import load_config

config =load_config()

def library_train_simulation(X,y,k,  outputs_dir='outputs/train/'):
    
    curr_vid_name = config['TRAIN']['curr_vid_name'][:-4]
    
    alpha = config['TRAIN']['alpha logisticTS']
    nchoices = y.shape[1]
    base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
    beta_prior = ((3./nchoices, 4), 2) # until there are at least 2 observations of each class, will use this prior
    beta_prior_ts = ((2./np.log2(nchoices), 4), 2)
    ### Important!!! the default values for beta_prior will be changed in version 0.3

    ## The base algorithm is embedded in different metaheuristics
    epsilon_greedy = EpsilonGreedy(deepcopy(base_algorithm), nchoices = nchoices,
                                beta_prior = beta_prior, random_state = 4444)
    adaptive_greedy_perc = AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices,
                                        decay_type='percentile', decay=0.9997,
                                        beta_prior=beta_prior, random_state = 7777)

                                        
    logistic_ts = LogisticTS(nchoices=nchoices, beta_prior=beta_prior_ts, random_state=5555, sample_from='coef', alpha=alpha)
    
    
    cloudpickle.dump(epsilon_greedy, open(f"ThompsonSampling/models/{curr_vid_name}/epsilon_greedy_model.pkl", "wb"))
    cloudpickle.dump(adaptive_greedy_perc, open(f"ThompsonSampling/models/{curr_vid_name}/adaptive_greedy_model.pkl", "wb"))
    cloudpickle.dump(logistic_ts, open(f"ThompsonSampling/models/{curr_vid_name}/logistic_ts_model.pkl", "wb"))
    
    
    
    
    
    
    #____ da qui solo per il grafico dell'accuracy____
      
    models = [epsilon_greedy, adaptive_greedy_perc, logistic_ts]

    # These lists will keep track of the rewards obtained by each policy
    rewards_egr, rewards_agr2, rewards_lts = [list() for i in range(len(models))]

    lst_rewards = [rewards_egr,rewards_agr2, rewards_lts]

    #predict for each model
    pred_egr, pred_agr2, pred_lts = [models[i].predict(X) for i in range(len(models))]
    lst_predicts = [pred_egr, pred_agr2, pred_lts]

    #decision function for each model
    dec_fun_egr, dec_fun_agr2, dec_fun_lts = [models[i].decision_function(X) for i in range(len(models))]
    lst_dec_fun = [dec_fun_egr, dec_fun_agr2, dec_fun_lts]

    # batch size - algorithms will be refit after N rounds
    batch_size =50 

    # initial seed - all policies start with the same small random selection of actions/rewards
    first_batch = X[:batch_size, :]
    np.random.seed(1)
    #import code; import code; code.interact(local=locals()) 

    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = y[np.arange(batch_size), action_chosen]

    # fitting models for the first time 
    for model in models:
        model.fit(X=first_batch, a=action_chosen, r=rewards_received)
        
    # these lists will keep track of which actions does each policy choose
    lst_a_egr, lst_a_agr2, lst_lts = [action_chosen.copy() for i in range(len(models))]
    lst_actions = [ lst_a_egr, lst_a_agr2, lst_lts]

    # rounds are simulated from the full dataset
    def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
        np.random.seed(batch_st)
        
        ## choosing actions for this batch
        actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
        
        # keeping track of the sum of rewards received
        rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
        
        # adding this batch to the history of selected actions
        new_actions_hist = np.append(actions_hist, actions_this_batch)
        
        # now refitting the algorithms after observing these new rewards
        np.random.seed(batch_st)
        model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist],
                warm_start = True)
        
        return new_actions_hist

    # now running all the simulation
    for i in range(int(np.floor(X.shape[0] / batch_size))):
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, X.shape[0]])
        
        for model in range(len(models)):
            lst_actions[model] = simulate_rounds(models[model],
                                                lst_rewards[model],
                                                lst_actions[model],
                                                X, y,
                                                batch_st, batch_end)
                
    generate_plot(batch_size=batch_size, lst_reward=lst_rewards, y=y)
    generate_hist(y=y, list_element=lst_actions, arr=k)
    generate_hist(y=y, list_element=lst_actions, arr=k, select_element=batch_size, name_plot='hist_last50_plot')
    generate_hist(y=y, list_element=lst_predicts, arr=k, select_element=batch_size, name_plot='hist_predict_plot')       


    library_result_df= pd.DataFrame({'X':[X], 'y':[y], 'egr_dec_func' : [dec_fun_egr], 'agr_dec_func':[dec_fun_agr2], 'lts-dec_func' : [dec_fun_lts] })
    library_result_df.to_csv(f'{outputs_dir}csv/lib_dec_func.csv' )   

    