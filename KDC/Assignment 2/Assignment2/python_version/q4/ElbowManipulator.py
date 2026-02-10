import numpy as np
from PKSubproblem import *


class ElbowManipulator:

    def __init__(self,l0,l1,l2):
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2

        # Calculate home position assuming robot is in 0 state in the given figure
        self.g_st = np.array([0,0,0,0,0,0,1.0])

        # Calculate twists
        self.twists = self.calculate_twists()

        # Calculate special points
        self.pb, self.qw, self.p = self.calculate_special_points()

        # Initialize PK Subproblem instance
        self.pk_solver = PKSolver()

    def calculate_twists(self):
        '''
        Docstring for calculate_twists

        :return: list of twists as 1D vector
        '''

        pass

    def calculate_special_points(self):
        '''
        Docstring for calculate_special_points
        
        :return: pb, qw, p : all 1D vectors of size 3
        '''

        pass

    def solve_ik(self, gd):
        '''
        Docstring for solve_ik
        
        :param gd: desired goal position

        :return q: list of joint angles
        '''

        # Implement the solution following the reference book

        pass

