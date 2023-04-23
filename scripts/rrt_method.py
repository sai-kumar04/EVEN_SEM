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

  ndof = 6
  planner = MyRRT(ndof)

  start = ob.State(planner.state_space)
  goal = ob.State(planner.state_space)

  

  p_task = np.array([[-0.932186], [-0.151], [0.6605]])
  p_task2 = np.array([[-0.932186], [-0.451], [0.6605]])
  path = []

  
  start_vector = inverse(p_task)
  goal_vector = inverse(p_task2, start_vector)
  for i in range(ndof):
      start[i] = start_vector[i]
      goal[i] = goal_vector[i]

  # ============== plan ===============
  print("REturned")
  result_path = planner.psolve(start, goal, 100)
  path = []
  joints = np.empty(ndof)
  for i in range(result_path.getStateCount()):
      
      for j in range(ndof):
          joints[j] = result_path.getStates()[i][j]
      path.append(copy.deepcopy(joints))
  print("All States Computed")
  print(path)

  allplots(path)

  sur_robot = RCMControl()

  print("Enter for forward")
  input()

  for i in path:
    sur_robot.move_to_q(i)
  sur_robot.stop()

    
    
    
  
    # moveit_commander.roscpp_initialize(sys.argv)
    # rospy.init_node('move_group_node', anonymous=True)

    # #robot = moveit_commander.RobotCommander()
    # #scene = moveit_commander.PlanningSceneInterface()
    # group_name = "manipulator"
    # move_group = moveit_commander.MoveGroupCommander(group_name)
    # print(move_group.get_named_targets())

    # for q in trajec:
    #   joint_goal = move_group.get_current_joint_values()
    #   joint_goal[0] = q[0]
    #   joint_goal[1] = q[1]
    #   joint_goal[2] = q[2]
    #   joint_goal[3] = q[3]
    #   joint_goal[4] = q[4]
    #   joint_goal[5] = q[5]
    #   joint_goal[6] = q[6]

    #   move_group.go(joint_goal, wait=True)
    

if __name__ == "__main__":
  main()