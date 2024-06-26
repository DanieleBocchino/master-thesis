import numpy as np
from skimage import measure

class Patch(object):
	
	def __init__(self, label, patch_num, value, a, center, axes, angle, area, boundaries, **kwargs):
        
		self.label = label
		self.patch_num = patch_num
		self.value = value
		self.a = a
		self.center = center
		self.axes = axes
		self.angle = angle
		self.area = area
		self.boundaries = boundaries
  
		for key,value in kwargs.items():
			if key == 'diag_size':
				self.diag_size = value
			if key=='tot_area':
				self.tot_area =value
			if key=='rho':
				self.rho =value
			if key=='expVal':
				self.expVal = value

	def compute_expected_reward(self, predictedMap):

		if self.label == 'Speaker' or self.label == 'non_Speaker':
			boundaries = self.boundaries[::10,:]
		else:
			boundaries = self.boundaries[::4,:]

		is_in = measure.grid_points_in_poly(predictedMap.shape, np.flip(boundaries,1))
	
		expectedRewPatch = np.sum(predictedMap[is_in])
		self.expect = self.value * expectedRewPatch
  
	