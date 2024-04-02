import math
class ContextualMAB:
    """
    Class to implement Contextual Multi Armed Bandit problem.
    """
   
    def __init__(self,mat_file, patches=[]):
        """
        Initialize the class.
        
        Parameters:
        mat_file : str
            The path of the file containing the data
        patches : list of tuples
            A list of patches in the form of (x, y, a, b, theta) where x, y are the coordinates of the center,
            a and b are the semi-major and semi-minor axes and theta is the angle of rotation
        """
        # we build len(patches) bandits
        self.mat_file = mat_file
        self.patches = patches

        


    def is_point_in_ellipse(self,patch_a, point):
        """
        Determine whether a given point is inside an ellipse defined by a patch.
        
        Parameters
        ----------
        self : object
            The object the function is being called on
        patch_a : list
            A list of parameters that define the ellipse, including the center coordinates (h, k), 
            semi-major and semi-minor axes (a, b), and rotation angle (theta)
        point : tuple
            A tuple of coordinates (x, y) representing the point to be checked

        Returns
        -------
        bool
            Returns True if the point is inside the ellipse, False otherwise
        """
        h,k = patch_a[0], patch_a[1]
        x,y = point
        a,b = patch_a[2]*1.5, patch_a[3]*1.5
        theta = patch_a[4]
            
        x_ = (x-h) * math.cos(theta) - (y-k) * math.sin(theta) 
        y_ = (x-h) * math.sin(theta) - (y-k) * math.cos(theta) 
            
        return (x_/a)**2 + (y_/b)**2 <=1

        
    def get_reward(self, k, patch_a, iframe): 
        """
        Calculates the reward, regret, and new current patch timestamp for a given point in an ellipse.
        
        Parameters
        ----------
        self : object
            The object the function is being called on
        k : int
            An integer representing the point in question
        patch_a : list
            A list of points that make up the patch
        iframe : int
            An integer representing the current timestamp

        Returns
        -------
        reward : int
            An integer, equal to 1 if the point is in the ellipse and 0 otherwise
        regret : int
            An integer, equal to 1 minus the reward
        new_curr_patch_ts : int
            The current timestamp of the patch
        """
        reward = 1 if self.is_point_in_ellipse(patch_a[k], self.mat_file[iframe])  else 0
        regret = 1-reward 
        new_curr_patch_ts = self.mat_file[iframe]
                    
        return reward, regret, new_curr_patch_ts
