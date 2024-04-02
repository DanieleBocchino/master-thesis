from matplotlib import pyplot as plt
from ThompsonSampling.CMAB import ContextualMAB
import pandas as pd
import math
from scipy.stats import norm as norm_dist
import numpy as np
from ThompsonSampling.OLR import OnlineLogisticRegression
from utils import load_config, rester_from_except

config = load_config()

class TS:   
    """
    The ThompsonSampler class is a class that implements Thompson Sampling algorithm, which is a Bayesian algorithm for balancing exploration and exploitation in multi-armed bandit problems.
    The class has the following attributes:

    k_list: A list to store the selected bandit (arm) at each round.
    reward_list: A list to store the rewards obtained from selecting the bandit.
    dist_dict_list: A list to store the distribution of the bandits.
    n_bandits: Number of bandits.
    curr_patch_ts: current position of the agent.
    round_df: Dataframe to store the information of each round, including the bandit selected, the context (x), the reward obtained, the regret, mean and variance of the bandit.
    analysis_df: Dataframe to store the information for analysis
    
    The class has the following methods:

    __init__(): The constructor method, it initializes all the attributes of the class.
    get_context(self, patches): This method takes in a list of patches and returns two lists, res and res2. res contains the context information of each patch (bias, id, distance and angle) and res2 contains the area of each patch.
    check_quality_of_patch(self, c_mab, patch_a, iframe): This method takes in an object of c_mab, patch_a and iframe and checks if the patch is inside the ellipse of the iframe. If the patch is inside the ellipse, it returns the index of the patch, else it returns None.
    TS_learning(self, patches,tsampling_lr, iframe, c_mab, analysis_df): This method implements the Thompson Sampling algorithm. It takes in a list of patches, an object of tsampling_lr, the current frame number, an object of c_mab and the analysis_df. It first gets the context and area of each patch using the get_context method. It then checks the quality of each patch using the check_quality_of_patch method. 
    """

    def __init__(self, lambda_:float, alpha:float, n_dim:int, n_bandits=2,  buffer_size=200,):
        self.k_list=[]
        self.reward_list = []
        self.dist_dict_list = []
        self.n_bandits=0
        self.curr_patch_ts = [0.0,0.0]
        self.round_df = pd.DataFrame({'k': [], 'x': [], 'reward': []})
        self.analysis_df = pd.DataFrame({'x': [], 'reward': []})
        self.contextual_library_df =  pd.DataFrame({'X': [], 'y': []})
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_bandits = n_bandits
        self.buffer_size = buffer_size
        self.n_dim = n_dim
        
    def fit_predict(self, data: pd.DataFrame, actual_x:np.ndarray)-> pd.DataFrame:
        """
        Fits the logistic regression model to the data and predicts the probability of success for the given input.
        
        Parameters:
        data : pd.DataFrame 
            The dataframe containing the input data and rewards.
        actual_x : np.ndarray
            The input for which the probability of success is to be predicted.
            
        Returns:
        pd.DataFrame
            Dataframe with columns 'prob', 'm' and 'q' which contains the predicted probability of success, the model parameters 'm' and 'q' respectively.
        """
                
        # sgd object
        olr = OnlineLogisticRegression(self.lambda_, self.alpha, self.n_dim)
                
        # fitting to data        
        olr.fit(np.vstack(data['x'].values), data['reward'].values)
                
        # data frame with probabilities and model parameters
        out_df = pd.DataFrame({'prob': olr.predict_proba(np.array(actual_x))[0][1],
                            'm': [olr.m] , 'q': [olr.q * (self.alpha) ** (-1.0)]})
        return out_df
    
    # decision function
    def choose_bandit(self, round_df: pd.DataFrame, actual_x:np.ndarray, iframe:int)->int:
        """
        Chooses the best bandit based on the predicted probability of success and returns it.
        
        Parameters:
        round_df : pd.DataFrame 
            The dataframe containing the data for the current round.
        actual_x : np.ndarray
            The input for which the probability of success is to be predicted.
        iframe : int
            The current frame number
            
        Returns:
        int
            The index of the best bandit.
        """
        
        # enforcing buffer size
        round_df = round_df.tail(self.buffer_size)

        if round_df.groupby(['k','reward']).size().shape[0] >=10: #10 è il massimo, oltre non entra mai >= 6: #>= 2 is minimum 
                        
            # predicting for two of our datasets
            self.ts_model_df = (round_df
                                .groupby('k')
                                .apply(self.fit_predict, actual_x=actual_x)
                                .reset_index().drop('level_1', axis=1).set_index('k'))
            
            # get best bandit
            best_bandit = int(self.ts_model_df['prob'].idxmax()) 
            
        # if we do not have, the best bandit will be random
        else:
            best_bandit = int(np.random.choice(list(range(self.n_bandits)),1)[0])
            self.ts_model_df = pd.DataFrame({'prob': 0.50, 'm': 0.0, 'q': self.lambda_}, index=[0])
            
        # return best bandit
        return best_bandit

    
    def get_context(self, patches, ):
        res, res2 = [], [] 
                
        angle_between = lambda p1, p2 : math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        
        #id --> CB = 0, STS = 1, NS=2, S=3
        for index,patch in enumerate(patches):
            id = 0 if patch.label == 'CB' else 1 if patch.label == 'STS' else 2 if patch.label == 'non_Speaker' else 3
            eDistance = math.dist(patch.center, self.curr_patch_ts)
            angle = angle_between(patch.center, self.curr_patch_ts)
            res.append(np.array([id,eDistance, angle]))
            res2.append(patch.a)

        return res, res2
    
    def check_quality_of_patch(self, c_mab, patch_a, iframe ):
        for i in range( len(patch_a)):
            if c_mab.is_point_in_ellipse(patch_a[i], c_mab.mat_file[iframe]):   
                return i
        return None
    
        
    def TS_learning(self, patches,  iframe, c_mab, analysis_df):
                
        # record information about this draw
        x, patch_a = self.get_context(patches=patches) # x = [bias, id, dist, angle]  e patch_a = [centre_x, centre_y, +/-centre_x, +/-centre_y, patch_angle]
        real_bandit = self.check_quality_of_patch(patch_a=patch_a, c_mab=c_mab, iframe=iframe)
    

        if(real_bandit):
            
            k = self.choose_bandit(self.round_df, x, iframe) #k is the best bandit

            y = []
            for i in range(len(patch_a)):
                y.append(1 if i==real_bandit else 0)
                                
            np_x = np.array(x).reshape(len(patches)*3,) 
            
            np_y = np.array(y)
        
            temp_analysis_df=pd.DataFrame({'x':[x], 'y': [y] })
            try:             
                temp_contextual_library_df = pd.DataFrame({'patch_label': 'CB' if int(x[k][0]) == 0 else 'STS' if int(x[k][0]) == 1 else 'non_Speaker' if int(x[k][0]) == 2 else 'Speaker' ,'X':[np_x], 'y': [np_y] }, index=[iframe])
            except:
                rester_from_except()

            reward, regret, new_curr_patch_ts = c_mab.get_reward(k, patch_a, iframe)
            self.curr_patch_ts = new_curr_patch_ts
    
            # record information about this draw
            self.k_list.append(k)
            self.reward_list.append(reward)

            temp_df = pd.DataFrame({'patch_label': 'CB' if int(x[k][0]) == 0 else 'STS' if int(x[k][0]) == 1 else 'non_Speaker' if int(x[k][0]) == 2 else 'Speaker' , 'x': [x[k]], 'k': int(k), 'reward': reward, 'regret' : regret, 'm' : [self.ts_model_df['m'][k if len(self.ts_model_df)!=1 else 0 ]], 'q' :[self.ts_model_df['q'][k if len(self.ts_model_df)!=1 else 0 ]]  }, index=[iframe])
        
            # accumulating in main df
            self.round_df = pd.concat([self.round_df, temp_df])
            self.analysis_df = pd.concat([self.analysis_df, temp_analysis_df])
            self.contextual_library_df = pd.concat([self.contextual_library_df, temp_contextual_library_df])

            return analysis_df
            

        