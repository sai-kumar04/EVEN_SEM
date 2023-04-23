#!/usr/bin/env python3


import sys
import rospy
from rrtct_planner import allplots

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics


import numpy as np

import tf_conversions.posemath as pm

#importing mpveit related
import moveit_msgs.msg


import moveit_commander
import geometry_msgs.msg

from math import pi 
from moveit_commander.conversions import pose_to_list

import matplotlib.pyplot as plt


class RCMControl:

  def __init__(self):

    self.dt = 0.05
    self.lam1 = 0
    self.p_task = None

    self.e1 = []
    self.e2 = []

    self.q0 = []
    self.q1 = []
    self.q2 = []
    self.q3 = []
    self.q4 = []
    self.q5 = []
    self.q6 = []
    #initializing moveit group
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_node', anonymous=True)

    #robot = moveit_commander.RobotCommander()
    #scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    print(move_group.get_named_targets())

    self.move_group = move_group




    #Initializing robot through URDF in parameter server

    self.robot = URDF.from_parameter_server('/robot_description')


    if self.robot != None:
      print("\n *** Successfully parsed urdf file and constructed kdl tree *** \n" )
    else:
      print("Failed to parse urdf file to kdl tree")

    self.tree = kdl_tree_from_urdf_model(self.robot)
    print(self.tree.getNrOfSegments())
    
    self.chain = self.tree.getChain("iiwa_link_0", "tcp")
    print(self.chain.getNrOfJoints())
    
    for body in self.robot.links:
      print(body.name)


    #robot chain till tip frame
    self.rcm_kin = KDLKinematics(self.robot, "iiwa_link_0", "tcp")
    
    #robot chain till iiwa_link_ee
    self.ee_kin = KDLKinematics(self.robot, "iiwa_link_0", "iiwa_link_ee")


  # def go_to_window(self):
  #   self.move_group.set_named_target("window1")
  #   success = self.move_group.go(wait=True)
  #   self.move_group.stop()
  #   self.move_group.clear_pose_targets()

  #   current_pose = self.move_group.get_current_pose().pose
  #   print(current_pose)
  def set_task(self, p_task):
    self.p_task = p_task

  def current_joint_angles(self):
    joint_angles = self.move_group.get_current_joint_values()
    return np.array(joint_angles)


  def getLambda(self, p_i, p_ip1, p_trocar):
    return np.linalg.norm(p_trocar - p_i)/np.linalg.norm(p_ip1 - p_i)


  def PosContImpl(self, q = None):
    #get current joint angles

    if np.size(q) == 0:
      q = self.rcm_kin.random_joint_angles()

    p_now = self.rcm_kin.forward(q)[:3, 3]
    print(p_now)

    
    J_ip1 = self.rcm_kin.jacobian(q)

    # print(J_ip1)

    J = J_ip1[:3]
    # print(J)

    J_pinv = np.linalg.pinv(J)

    print(J_pinv)

    # E = np.zeros([3, 1])
    # print(p_now)
    # print(p_task)
    E = p_task - p_now
    print(E)
    if np.linalg.norm(E) < 0.00001:
      return None

    K = np.array([[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])


    dq = np.matmul(np.matmul(J_pinv, K), E)
    print(dq)
    print(q)

    q_new = q + self.dt*np.transpose(dq)

    print(q_new)
    print(q_new[0,1], q_new[0,2])
    return [q_new[0,0], q_new[0, 1], q_new[0, 2], q_new[0, 3], q_new[0, 4], q_new[0, 5], q_new[0, 6]]

  def RCMImpl(self, q):

    p_task = self.p_task

    p_i = self.ee_kin.forward(q)[:3, 3]
    # print("p_i: ", p_i)

    p_ip1 = self.rcm_kin.forward(q)[:3, 3]
    print("p_ip1: ", p_ip1)

    AP = np.array(p_trocar - p_i)
    d = np.array(p_ip1 - p_i)


    lam = np.dot(np.transpose(AP), d)[0,0]/np.linalg.norm(d)**2

    #get lambda
    # lam = self.getLambda(p_i, p_ip1, p_trocar)
    # print("lam: ", lam)
    # lam = self.lam1

    J_i = self.ee_kin.jacobian(q)[:3]
    J_ip1 = self.rcm_kin.jacobian(q)[:3]

    # print(J_ip1)

    p_rcm = p_i + lam*(p_ip1 - p_i)
    # print("p_rcm: ", p_rcm)

    J_rcm = np.zeros([3, 7])
    J_rcm[:, :7] = (J_i + lam*(J_ip1 - J_i))[:3, :]
    J_rcm[:, 7:] = p_ip1 - p_i
    print("J_rcm: ", J_rcm)

    J_T = J_ip1
    

    #Extended Jacobian
    J_ext = np.zeros([6, 7])
    J_ext[:3, :7] = J_T
    J_ext[3: , :] = J_rcm
    print("J_ext: ", J_ext)

    

    J_pinv = np.linalg.pinv(J_ext)

    

    E = np.zeros([6, 1])
    
    E[:3,:] = p_task - p_ip1
    E[3:, :] = p_trocar - p_rcm
    print("E: ", E)
    print(np.linalg.norm(E))
    if np.linalg.norm(E) < 0.007:
      return None

    K = np.array([[1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])
    self.e1.append(np.linalg.norm(p_task - p_ip1))
    self.e2.append(np.linalg.norm(p_trocar - p_rcm))


    dq = np.matmul(np.matmul(J_pinv, K), E)
    

    q_new = q + self.dt*np.transpose(dq[:7,:])

    self.q0.append(q_new[0,0])
    self.q1.append(q_new[0,1])
    self.q2.append(q_new[0,2])
    self.q3.append(q_new[0,3])
    self.q4.append(q_new[0,4])
    self.q5.append(q_new[0,5])
    self.q6.append(q_new[0,6])
    return [q_new[0,0], q_new[0, 1], q_new[0, 2], q_new[0, 3], q_new[0, 4], q_new[0, 5], q_new[0, 6]]



  def move_to_q(self, q):

    joint_goal = self.move_group.get_current_joint_values()
    joint_goal[0] = q[0]
    joint_goal[1] = q[1]
    joint_goal[2] = q[2]
    joint_goal[3] = q[3]
    joint_goal[4] = q[4]
    joint_goal[5] = q[5]
    joint_goal[6] = q[6]

    self.move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    # self.move_group.stop()

  def move_to_trcr(self):

    

    
    joint_goal = self.move_group.get_current_joint_values()
    joint_goal[0] = 0.1
    joint_goal[1] = 0.6941
    joint_goal[2] = 0.0000
    joint_goal[3] = 1.2947
    joint_goal[4] = 0.000
    joint_goal[5] = -1.3995
    joint_goal[6] = 0.000

    self.move_group.go(joint_goal, wait=True)
    self.move_group.stop()




  def stop(self):
    self.move_group.stop()

  def get_plts(self):
    plt.figure(1)
    plt.plot(self.e1)
    plt.xlabel("Iteration number")
    plt.ylabel("Task Space Error")

    plt.figure(2)
    plt.plot(self.e2)
    plt.xlabel("Iteration number")
    plt.ylabel("RCM Error")
    
    plt.figure(3)
    plt.plot(self.q0)
    plt.plot(self.q1)
    plt.plot(self.q2)
    plt.plot(self.q3)
    plt.plot(self.q4)
    plt.plot(self.q5)
    plt.plot(self.q6)
    plt.xlabel("Iteration number")
    plt.ylabel("Joint Angles")
    plt.legend(["Jt1", "Jt2", "Jt3", "Jt4", "Jt5", "Jt6", "Jt7"])

    plt.show()

    

p_trocar = np.array([[-0.63218696], [0.0000982], [0.760526907]])
p_task = np.array([[-0.932186], [-0.151], [0.6605]])
p_task2 = np.array([[-0.932186], [-0.451], [0.6605]])

def main():
  
  
  print("lm0:", 1 - np.linalg.norm(p_trocar-p_task)/0.61)

  sur_robot = RCMControl()
  
  input()

  print("press Enter to continue")

  sur_robot.move_to_trcr()
  print("Enter for forward")
  input()
  path = []
  
  

  for j in range(20):
    p_temp = p_trocar + (-p_trocar + p_task)*(j+1)/20
    sur_robot.set_task(p_temp)
    print("j:", j)
    angles = sur_robot.current_joint_angles()
    print(angles)
    for i in range(1000):
      print("Iter: ", i)
      
      ange = sur_robot.RCMImpl(angles)
      if ange == None:
        sur_robot.stop()
        print("Destination reached")
        break

      print(ange)
      angles = ange
      # sur_robot.move_to_q(ange)

    sur_robot.move_to_q(angles)
    path.append(angles)

  for j in range(20):
    p_temp = p_task + (-p_task + p_task2)*(j+1)/20
    sur_robot.set_task(p_temp)
    print("j:", j)
    angles = sur_robot.current_joint_angles()
    print(angles)
    for i in range(1000):
      print("Iter: ", i)
      
      ange = sur_robot.RCMImpl(angles)
      if ange == None:
        sur_robot.stop()
        print("Destination reached")
        break

      print(ange)
      angles = ange
      # sur_robot.move_to_q(ange)

    sur_robot.move_to_q(angles)
    path.append(angles)
  allplots(path)

  

  

  

  # sur_robot.get_plts()

  




  


  
  

if __name__ == "__main__":
  main()







