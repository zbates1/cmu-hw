from ElbowManipulator import *
import numpy as np

l0 = 0.3
l1 = 0.5
l2 = 0.6

gd = np.array([0.6,0.4,0.0,1.0,0,0,0.0]) # x,y,z,qx,qy,qz,qw

manipulator = ElbowManipulator(l0,l1,l2)

q = manipulator.solve_ik(gd)

print(f"Solution to the IK problem: {q}")

