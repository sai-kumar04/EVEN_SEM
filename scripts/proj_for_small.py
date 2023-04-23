#!/usr/bin/env python3

import sys
import copy

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from rrtct_planner import MyRRT
from rrtct_planner import inverse, allplots
import numpy as np


import rospy
import moveit_commander
import moveit_msgs.msg

import math
import copy
import numpy as np
from numpy import linalg as LA
from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

f = open('/home/sai_kumar/iiwa7_setup/src/iiwa7_with_tool/urdf/iiwa7_with_tool.urdf', 'r')
robot = URDF.from_xml_string(f.read())  # parsed URDF
ee_kin = KDLKinematics(robot, "iiwa_link_0", "tcp")
ur_kin = KDLKinematics(robot, "iiwa_link_0", "iiwa_link_ee")
# p_trocar = np.array([[ 9.10937772e-01],[-1.31414659e-12],[ 6.54741958e-01]])
p_trocar = np.array([[-0.63218696], [0.0000982], [0.760526907]])
p_task = np.array([[-0.932186], [-0.151], [0.6605]])
p_task2 = np.array([[-0.932186], [-0.451], [0.6605]])

def constraintfunction(q = ur_kin.random_joint_angles()):
    
    out = np.zeros((3,1), dtype=np.float64)
    
    
    p_i = ur_kin.forward(q)[:3, 3]
    p_t = ee_kin.forward(q)[:3, 3]

    AP = p_trocar - p_i
    
    d = p_t - p_i
    

    # dist = LA.norm(np.cross(np.transpose(AP), np.transpose(d)))/LA.norm(d)
    lam = np.dot(np.transpose(AP), d)[0,0]/LA.norm(d)**2
    p_rcm = p_i + lam*(p_t - p_i)
    # print("P_rcm: ",p_rcm)

    out[0] = (p_rcm - p_trocar)[0,:]
    out[1] = (p_rcm - p_trocar)[1,:]
    out[2] = (p_rcm - p_trocar)[2,:]

    
    return out


def jacobian(q = ur_kin.random_joint_angles()):
        
    out = np.zeros((3, 7), dtype=np.float64)
    
    
    
    p_i = ur_kin.forward(q)[:3, 3]
    p_t = ee_kin.forward(q)[:3, 3]

    AP = np.array(p_trocar - p_i)
    d = np.array(p_t - p_i)


    lam = np.dot(np.transpose(AP), d)[0,0]/LA.norm(d)**2
    J_i = ur_kin.jacobian(q)[:3]
    J_t = ee_kin.jacobian(q)[:3]


    p_rcm = p_i + lam*(p_t - p_i)
    J_rcm = J_i + lam*(J_t - J_i)

    # Dcap = p_trocar - p_rcm

    # J = np.dot(np.transpose(Dcap), J_rcm)
    # nrm = np.linalg.norm(J)
    # if np.isfinite(nrm) and nrm > 0:
    #     out[0, :] = [J[0, 0], J[0, 1], J[0, 2], J[0, 3], J[0, 4], J[0, 5], J[0, 6]]
    # else:
    #     out[0, :] = [1, 0, 0, 0, 0, 0, 0]
    out[0] = (J_rcm)[0,:]
    out[1] = (J_rcm)[1,:]
    out[2] = (J_rcm)[2,:]

    return out


def projectionFunction(q =ur_kin.random_joint_angles()):
    # print("q: ", q)
    fx = constraintfunction(q)
    

    while np.linalg.norm(fx) > 0.005:
        jc = jacobian(q)
        invj = np.linalg.pinv(jc)
        dq = np.matmul(invj, fx)
        dq = np.transpose(dq)[0]
        q = q - dq
        # print(q)
        
        
        

        fx = constraintfunction(q)
        # print(np.linalg.norm(fx))

    return q


def RCMjacobian(q=ur_kin.random_joint_angles()):
        
    out = np.zeros((6, 7), dtype=np.float64)
    
    
    
    p_i = ur_kin.forward(q)[:3, 3]
    p_t = ee_kin.forward(q)[:3, 3]

    AP = np.array(p_trocar - p_i)
    d = np.array(p_t - p_i)


    lam = np.dot(np.transpose(AP), d)[0,0]/LA.norm(d)**2
    J_i = ur_kin.jacobian(q)[:3]
    J_t = ee_kin.jacobian(q)[:3]


    p_rcm = p_i + lam*(p_t - p_i)
    J_rcm = J_i + lam*(J_t - J_i)

    # Dcap = p_trocar - p_rcm

    # J = np.dot(np.transpose(Dcap), J_rcm)
    # nrm = np.linalg.norm(J)
    # if np.isfinite(nrm) and nrm > 0:
    #     out[0, :] = [J[0, 0], J[0, 1], J[0, 2], J[0, 3], J[0, 4], J[0, 5], J[0, 6]]
    # else:
    #     out[0, :] = [1, 0, 0, 0, 0, 0, 0]
    out[0] = (J_t)[0,:]
    out[1] = (J_t)[1,:]
    out[2] = (J_t)[2,:]
    out[3] = (J_rcm)[0,:]
    out[4] = (J_rcm)[1,:]
    out[5] = (J_rcm)[2,:]

    return out


def RCMImpl(q, p_ts):
        
    J_ext = np.zeros((6, 7), dtype=np.float64)
    
    
    
    p_i = ur_kin.forward(q)[:3, 3]
    p_t = ee_kin.forward(q)[:3, 3]

    AP = np.array(p_trocar - p_i)
    d = np.array(p_t - p_i)


    lam = np.dot(np.transpose(AP), d)[0,0]/LA.norm(d)**2
    J_i = ur_kin.jacobian(q)[:3]
    J_t = ee_kin.jacobian(q)[:3]


    p_rcm = p_i + lam*(p_t - p_i)
    J_rcm = J_i + lam*(J_t - J_i)

    E = np.zeros([6, 1])

    E[:3,:] = p_ts- p_t
    E[3:, :] = p_trocar - p_rcm

    if np.linalg.norm(p_ts- p_t) < 0.0001 and np.linalg.norm(p_trocar - p_rcm)<0.005:
            return None

    J_ext[0] = (J_t)[0,:]
    J_ext[1] = (J_t)[1,:]
    J_ext[2] = (J_t)[2,:]
    J_ext[3] = (J_rcm)[0,:]
    J_ext[4] = (J_rcm)[1,:]
    J_ext[5] = (J_rcm)[2,:]

    J_pinv = np.linalg.pinv(J_ext)
    dq = np.matmul(J_pinv, E)
    dq = np.transpose(dq)[0]

    return dq




class RCMControl:

  def __init__(self):

        #initializing moveit group
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_node', anonymous=True)

    #robot = moveit_commander.RobotCommander()
    #scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    print(move_group.get_named_targets())

    self.move_group = move_group

  def move_to_q(self, q):

    joint_goal = self.move_group.get_current_joint_values()
    joint_goal[0] = q[0]
    joint_goal[1] = q[1]
    joint_goal[2] = q[2]
    joint_goal[3] = q[3]
    joint_goal[4] = q[4]
    joint_goal[5] = q[5]
    joint_goal[6] = 0

    self.move_group.go(joint_goal, wait=True)


  def stop(self):
    self.move_group.stop()


def main():

    q1 = np.array([0.1, 0.6941, 0.0000, 1.2947, 0.000, -1.3995, 0.000])

    reached = False
    while not reached:
        dq = RCMImpl(q1, p_task)
        if dq is None:
            reached = True
            break
        q1 = q1 + 0.1*dq
    print(q1)
    q2 = q1
    reached = False
    while not reached:
        dq = RCMImpl(q2, p_task2)
        if dq is None:
            reached = True
            break
        q2 = q2 + 0.1*dq

    print(q2)

    print(np.linalg.norm(q2-q1))
    q_proj = []
    q = q1
    i = 0
    while np.linalg.norm(ee_kin.forward(q1)[:3, 3] -ee_kin.forward(q2)[:3, 3]) > 0.0001:
        q = q1 + (q2 - q1)/np.linalg.norm(q2 - q1)*0.001
        q1 = projectionFunction(q)
        q_proj.append(q1)
        i = i + 1
        print(i)
    print(q_proj)

    allplots(q_proj)





  

    

if __name__ == "__main__":
  main()