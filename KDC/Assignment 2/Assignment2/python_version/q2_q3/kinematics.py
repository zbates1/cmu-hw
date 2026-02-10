import numpy as np

class WarretArm:
    def __init__(self):
        ## Define required twists and/or matrices from the information given about the arm and its position in 3D space

        pass

    def ForwardKinematics(self, q):
        '''
        q2.1


        :param q: joint angles

        :return outputs: Pose of the end effector wrt to the world frame
                         expressed as 1D vector of position and quaterion
                         (px,py,pz,qx,qy,qz,qw)
        '''

        pass

    def ComputeJacobian(self, q):
        '''
        q3.1
        Calculate arm jacobian
        
        :param q: joint angles

        :return J: a 6x7 ndarray
        '''

        pass

    def ComputeEEVel(self, xs, xd):
        '''
        q3.2
        Calculate end effector velocity
        
        :param xs: starting end effetor pose as 1D vector of size 7
        :param xd: desired end effetor pose as 1D vector of size 7

        :return v: a 1D array of 6 elements, linear followed by angular
        '''

        pass

    def IK_PseudoInverse(self, q_guess, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        q3.3
        Iterative method of solving using pseudo inverse approach

        :param  q_guess: joint angles
                TGoal: Desired end effector pose as 1D vector of 7 elements
                x_eps: euclidian error in position
                r_eps: euclidian error in se(2) (norm of axis angle representation of error)

        :return q: computed joint angles to achieve desired end effector pose, 
        '''

        pass

    def IK_LeastSquares(self, q_guess, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        q3.4
        Optimization method of solving using damped least squares

        :param  q_guess: joint angles
                TGoal: Desired end effector pose as 1D vector of 7 elements
                x_eps: euclidian error in position
                r_eps: euclidian error in se(2) (norm of axis angle representation of error)

        :return q: computed joint angles to achieve desired end effector pose, 
        '''

        pass


def GetWhiteboardPose(data):
    '''
    q2.2
    Find out where the whiteboard is
    
    :param data: define as seen fit by you

    :return centroid, normal: both are 1D vectors with 3 elements
    '''

    pass


