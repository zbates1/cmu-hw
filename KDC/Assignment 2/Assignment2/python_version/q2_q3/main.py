from kinematics import *
import numpy as np

jointdata_file = "../../JointData.txt"

arm = WarretArm()


### q2 ###

# Export file names
marker_tip_traj_filename = "../../MarkerTipData.txt"
animation_filename = "../../RobotWriting.mp4"

## Import JointData.txt

## Calculate marker tip trajectory
# x = arm.FrorwardKinematics(q)

## Export marker tip trajectory and animation (optional)

## Calculate Whiteboard pose
position, normal = GetWhiteboardPose(
    # Send required args
)

print(f"Question 2.2: Whiteboard position and orientation")
print(f"position: {position}")
print(f"normal: {normal}")




### q3 ###
theta_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
x_s = np.array([0.44543, 1.12320, 2.22653, -0.29883, 0.44566, 0.84122, -0.06664])
x_d = np.array([0.46320, 1.16402, 2.22058, -0.29301, 0.41901, 0.84979, 0.12817])
x_d1 = x_d
x_d2 = np.array([0.49796, 0.98500, 2.34041, -0.11698, 0.07755, 0.82524, 0.54706])

## q3.1 Jacobian
J = arm.ComputeJacobian(theta_s)

print(f"Question 3.1: Computing Jacobian")
print(f"Output velocities are ...") # @TODO Explain your jacobian representation. Include frames and velocity formats
print(f"J: \n {J}")

## q3.2 Velocity
v = arm.ComputeEEVel(x_s, x_d)

print(f"Question 3.2: Computing Velocity")
print(f"Velocity is ...") # @TODO Explain your velocity representation. Include frames and formats
print(f"v: \n {v}")

## q3.3 Computing IK 1
q = arm.IK_PseudoInverse(theta_s, x_d1)

print(f"Question 3.3: Computing IK using Pseudo Inverse")
print(f"q: \n {q}")

## q3.3 Computing IK 2
q = arm.IK_LeastSquares(theta_s, x_d1)

print(f"Question 3.3: Computing IK using Least Sqaures")
print(f"q: \n {q}")