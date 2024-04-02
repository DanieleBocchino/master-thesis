import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from ThompsonSampling.BOF import BoF
import pickle
from ThompsonSampling.Library_Test import library_test_simulation
from utils import load_config

config =load_config()

def logistic(beta, dist):
	return 1/(1+np.exp(-beta*dist))

def sample_OU_trajectory(numOfpoints, alpha, mu, startxy, dt, sigma):

	Sigma1 = np.array([[1, np.exp(-alpha * dt / 2)],[np.exp(-alpha * dt / 2), 1]])
	Sigma2 = Sigma1

	x = np.zeros(numOfpoints)
	y = np.zeros(numOfpoints)
	mu1 = np.zeros(numOfpoints)
	mu2 = np.zeros(numOfpoints)
	x[0] = startxy[1]
	y[0] = startxy[0]
	mu1[0] = mu[1]
	mu2[0] = mu[0]
	for i in range(numOfpoints-1):
		r1 = np.random.randn() * sigma
		r2 = np.random.randn() * sigma

		x[i+1] = mu1[i]+(x[i]-mu1[i])*(Sigma1[0,1]/Sigma1[0,0])+np.sqrt(Sigma1[1,1])-(((Sigma1[0,1]**2)/Sigma1[0,0]))*r1
		y[i+1] = mu2[i]+(y[i]-mu2[i])*(Sigma2[0,1]/Sigma2[0,0])+np.sqrt(Sigma2[1,1])-(((Sigma2[0,1]**2)/Sigma2[0,0]))*r2
		mu1[i+1] = mu1[i]
		mu2[i+1] = mu2[i]

	return np.stack([y,x], axis=1)

def get_context(patches, curr_patch):
	X= [] 
	angle_between = lambda p1, p2 : math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        
	#id --> CB = 0, STS = 1, NS=2, S=3
	for index,patch in enumerate(patches):
		id = 0 if patch.label == 'CB' else 1 if patch.label == 'STS' else 2 if patch.label == 'non_Speaker' else 3
		eDistance = math.dist(patch.center,curr_patch)
		angle = angle_between(patch.center, curr_patch)
		X.append(np.array([id,eDistance, angle]))

	return np.array(X)

class GazeSampler(object):  
	def __init__(self, video_fr, phi, kappa, images =[]):        		
		self.allFOA = []
		self.sampled_gaze = []
		self.fly = False
		self.eps = np.finfo(np.float32).eps
		self.video_fr = video_fr
		self.phi = phi
		self.kappa = kappa
		self.images = images
		self.gif_df = pd.DataFrame()
  

# ------------------------------------------------ GAZE SAMPLING WITH FORAGING STUFF -------------------------------------------------------------------------------


	def sample(self, iframe, patches, FOAsize, seed=None, verbose=False, debug=False):
     
		gp, Q, dist, jump_prob, curr_rho, A = 1, 0, 1, 0, 0, 0

		if seed is not None:
			np.random.seed(seed)

		self.patch = patches
		self.FOAsize = FOAsize

		numPatch = len(self.patch)
		if iframe == 0:				#first frame
			
			self.s_ampl = []
			self.s_dir = []
			self.fix_dur = []
			self.fix_pos = []

			z = 0			#start from the CB
			self.z_prev = 0
			m = self.patch[0].center
			s = np.asarray(self.patch[0].axes)
			S = np.eye(2) * s
			arrivingFOA = np.random.multivariate_normal(m, S)
			
			self.allFOA.append(arrivingFOA)
			self.newFOA = arrivingFOA
			prevFOA = arrivingFOA
			self.curr_fix_dur = 0

		else:
			if self.z_prev >= numPatch:
				self.z_prev = np.random.choice(numPatch, 1, p=np.ones(numPatch)/numPatch)[0]
			
			#curr_value = self.patch[self.z_prev].value

			curr_patch = self.patch[self.z_prev].center #VA CAMBIATO PROBABILMENTE!!!!!!!

			X = get_context(patches=patches, curr_patch=curr_patch)
			codebook = np.load(f"{config['TRAIN']['models_dir']}codebook.npy")
   
			X_test_bof, _ = BoF(X, codebook=codebook, k=25, normalize=True)	
   
   
			#rbf_features = np.load('ThompsonSampling/models/rbf_features.npy', allow_pickle=True)
			# Load the sampler object from the file
			filename = f"{config['TRAIN']['models_dir']}rbf_features.pkl"
			with open(filename, "rb") as file:
				rbf_features = pickle.load(file)
   
			#import code; code.interact(local=dict(globals(), **locals()))
			X_features = rbf_features.transform(X_test_bof)
   
			dec_fun_egr, dec_fun_agr2, dec_fun_lts = library_test_simulation(X_features)
			curr_rho = dec_fun_lts[self.z_prev]
			others_rho = dec_fun_lts
   
			others_rho = np.delete(others_rho, self.z_prev )

			Q = np.mean(others_rho)		
			
			A = self.phi / (curr_rho+self.eps) 			#A = self.phi / (curr_value+self.eps)

			#posso tunare la curva sostituendo curr_rho con una costante
			gp = self.kappa * np.exp(-A*self.curr_fix_dur)
			dist = gp - Q																#distance of the current gut-f value from the average patch value
			jump_prob = 1 - logistic(config['TEST CONFIG']['logistic'], dist) #tuning sul 20 
			
			rand_num = np.random.rand()
			
			if debug:
				print('\nCurrent Values for patches:')
				#print([patch.value for patch in self.patch])
				print('\nInstantaneous Reward Rate: ' + str(gp))
				#print('Current value for the present patch: ' + str(curr_value))
				print('Current mean value for other patches: ' + str(Q))
				print('Distance between threshold and GUT: ' + str(dist))
				print('Jump Probability: ' + str(jump_prob))
				print('Random Number: ' + str(rand_num) + '\n')
				
				import code
				code.interact(local=locals())

			if rand_num < jump_prob:
				#possible saccade
				rho =dec_fun_lts # np.array([patch.compute_rho(np.array(self.patch[self.z_prev].center), True, self.kappa) for patch in self.patch])
				pi_prob = rho / np.sum(rho)		
				#import code; code.interact(local=dict(globals(), **locals()))
				z = np.random.choice(numPatch, 1, p=pi_prob)[0]	#choose a new patch

				if debug:
					print('\npi_prob: ')
					print(pi_prob)
					print('Choice:')
					print(z)

					import code
					code.interact(local=locals())

				rand_num2 = np.random.rand()		#for the random choice (small saccade of keep fixating)

				if z != self.z_prev or rand_num2 > 0.6:		#increase the value on the rhs to decrease the number of small saccades
					# if here, then a new patch has been choosen, OR a small saccade will take place
					self.z_prev = z
					m = self.patch[z].center
					s = np.asarray(self.patch[z].axes) * 3
					S = np.eye(2) * s
					self.newFOA = np.random.multivariate_normal(m, S)	#choose position inside the (new) patch
					prevFOA = self.allFOA[-1]
				else:
					# if here, will keep fixating the same patch
					z = self.z_prev 
					prevFOA = self.allFOA[-1]
					self.newFOA = prevFOA
			else:
				# keep fixating...
				z = self.z_prev 
				prevFOA = self.allFOA[-1]
				self.newFOA = prevFOA

		#timeOfFlight = euclidean(self.newFOA, prevFOA)
		timeOfFlight = np.linalg.norm(self.newFOA - prevFOA)

		#Possible FLY --------------------------------------------------------------------------------------------------------------------------------------------
		if timeOfFlight > self.FOAsize:	
			if verbose:
				print('\nnew and prev: ', self.newFOA, prevFOA)
				#different patch from current
				print('\n\tFLY!!!! New patch: ' + self.patch[z].label)
			self.FLY = True
			self.SAME_PATCH = False
			self.allFOA.append(self.newFOA)

			self.s_ampl.append(timeOfFlight)
			curr_sdir = np.arctan2(self.newFOA[1]-prevFOA[1], self.newFOA[0]-prevFOA[0]) + np.pi
			self.s_dir.append(curr_sdir)
			self.fix_dur.append(self.curr_fix_dur)
			self.fix_pos.append(prevFOA)
			self.curr_fix_dur = 0
		else:
			self.FLY = False
			self.SAME_PATCH = True
			self.curr_fix_dur += 1/self.video_fr

		#FEED ---------------------------------------------------------------------------------------------------------------------------------------------------
		if self.SAME_PATCH == True and iframe > 0:
			#if we are approximately within the same patch retrieve previous exploitation/feed state and continue the local walk... 
			if verbose:
				print('\n\tSame Patch, continuing exporation...')
			startxy = self.xylast
			#t_patch = self.t_patchlast
			alphap = self.alphaplast                          
			mup = self.newFOA
			#dtp = self.dtplast
			sigmap = self.sigma_last

		elif self.SAME_PATCH == False or iframe == 0:
			#...init new exploitation/feed state for starting a new random walk   
			if verbose:        
				print('\n\tExploring new Patch...')

			if iframe == 0:
				startxy = self.newFOA
			else: 
				startxy = self.sampled_gaze[-1][-1,:]

			mup = self.newFOA

			if not self.FLY:
				alphap = np.max([self.patch[z].axes[0], self.patch[z].axes[1]])*10
				sigmap = 1
			else:
				alphap = timeOfFlight/5.
				sigmap = 15

			self.alphaplast = alphap
			self.sigma_last = sigmap
					
		walk = sample_OU_trajectory(numOfpoints=10, alpha=alphap, mu=mup, startxy=startxy, dt=1/alphap, sigma=sigmap)
		
		arrivingFOA = np.round(walk[-1,:])
		self.xylast = arrivingFOA
		#self.muplast = mup
		self.sampled_gaze.append(walk)
  
		df2=pd.DataFrame({ 'gp' : gp, 'curr_fix_dur':self.curr_fix_dur, 'curr_rho':curr_rho, 'A':A,   'Q' : Q, 'dist' : dist, 'jump_prob' : jump_prob }, index=[iframe])
		self.gif_df =pd.concat([self.gif_df, df2])



		
