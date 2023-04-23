#!/usr/bin/env python3

import sys

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import math
import copy
import numpy as np
from numpy import linalg as LA
from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open('/home/sai_kumar/iiwa7_setup/src/iiwa7_with_tool/urdf/iiwa7_with_tool.urdf', 'r')
robot = URDF.from_xml_string(f.read())  # parsed URDF
ee_kin = KDLKinematics(robot, "iiwa_link_0", "tcp")
ur_kin = KDLKinematics(robot, "iiwa_link_0", "iiwa_link_ee")
# p_trocar = np.array([[ 9.10937772e-01],[-1.31414659e-12],[ 6.54741958e-01]])
p_trocar = np.array([[-0.63218696], [0.0000982], [0.760526907]])


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

def inverse(p_task, q_near = np.array([0.1, 0.6941, 0.0000, 1.2947, 0.000, -1.3995, 0.000])):
    q1 = q_near

    reached = False
    while not reached:
        dq = RCMImpl(q1, p_task)
        if dq is None:
            reached = True
            break
        q1 = q1 + 0.1*dq
    return q1

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

def allplots(path):

    ee_list  = []
    for i in path:
        print(i)
        q = [i[0],i[1],i[2],i[3],i[4],i[5],0.00]
        e = ee_kin.forward(q)[:3, 3]
        ee_list.append([e[0,0], e[1,0], e[2,0]])

    print(ee_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_data = []
    y_data = []
    z_data = []
    for i in ee_list:
        x_data.append(i[0])
        y_data.append(i[1])
        z_data.append(i[2])
    
    ax.scatter3D(x_data[0],y_data[0],z_data[0], 'red',s=50)
    ax.scatter3D(x_data[-1],y_data[-1],z_data[-1], 'green',s=50)

    ax.plot3D(x_data,y_data,z_data, 'gray')
    ax.scatter3D(x_data[1:-1],y_data[1:-1],z_data[1:-1], 'red')
    ax.set_title('Generated Surgical Tool Tip Path')
    ax.set_xlabel('X (in m)')
    ax.set_ylabel('Y (in m)')
    ax.set_zlabel('Z (in m)')

    plt.show()

    rcm_list  = []
    for i in path:
        q = [i[0],i[1],i[2],i[3],i[4],i[5],0.00]
        e = constraintfunction(q)
        rcm_list.append([e[0,0], e[1,0], e[2,0]])

    print(rcm_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_data = []
    y_data = []
    z_data = []
    for i in rcm_list:
        x_data.append(i[0])
        y_data.append(i[1])
        z_data.append(i[2])
    # ax.plot3D(x_data,y_data,z_data, 'gray')
    ax.scatter3D(x_data,y_data,z_data, 'red')

    r = 0.005
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = r*np.cos(u) * np.sin(v)
    y = r*np.sin(u) * np.sin(v)
    z = r*np.cos(v)
    ax.plot_surface(x, y, z, alpha = 0.1, color='purple')

    ax.set_title('Remote Center Motion Position')
    ax.set_xlabel('X (in m)')
    ax.set_ylabel('Y (in m)')
    ax.set_zlabel('Z (in m)')

    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    i = 0
    x_data0 = [ee_list[i][0], rcm_list[i][0]]
    y_data0 = [ee_list[i][1], rcm_list[i][1]]
    z_data0 = [ee_list[i][2], rcm_list[i][2]]
    ax.plot3D(x_data0,y_data0,z_data0, 'red', linewidth=4)
    for i in range(len(ee_list)):
        x_data = [ee_list[i][0], rcm_list[i][0]]
        y_data = [ee_list[i][1], rcm_list[i][1]]
        z_data = [ee_list[i][2], rcm_list[i][2]]
        ax.plot3D(x_data,y_data,z_data, 'gray')
    i = len(ee_list) - 1
    x_data1 = [ee_list[i][0], rcm_list[i][0]]
    y_data1 = [ee_list[i][1], rcm_list[i][1]]
    z_data1 = [ee_list[i][2], rcm_list[i][2]]
    ax.plot3D(x_data1,y_data1,z_data1, 'green',linewidth=4)

    ee_list=[]
    for i in path:
        print(i)
        q = [i[0],i[1],i[2],i[3],i[4],i[5],0.00]
        e = ee_kin.forward(q)[:3, 3]
        ee_list.append([e[0,0], e[1,0], e[2,0]])

    print(ee_list)

    x_data = []
    y_data = []
    z_data = []
    for i in ee_list:
        x_data.append(i[0])
        y_data.append(i[1])
        z_data.append(i[2])
    
    ax.scatter3D(rcm_list[0][0],rcm_list[0][1],rcm_list[0][2], 'blue', s=100 , alpha = 0.7)
    # ax.scatter3D(x_data[-1],y_data[-1],z_data[-1], 'green')

    ax.plot3D(x_data,y_data,z_data, 'blue',linewidth=4)
    # ax.scatter3D(x_data[1:-2],y_data[1:-2],z_data[1:-2], 'red')

    ax.set_title('Surgical Tool Representation through the Path')
    ax.set_xlabel('X (in m)')
    ax.set_ylabel('Y (in m)')
    ax.set_zlabel('Z (in m)')

    plt.show()



class MyStateSpace(ob.RealVectorStateSpace):
    def __init__(self, ndof):
        self.ndof = ndof
        super(MyStateSpace, self).__init__(self.ndof)

        lower_limits = [-2.9321, -2.0594, -2.9321, -2.0594,-2.9321, -2.0594]
        upper_limits = [2.9321, 2.0594, 2.9321, 2.0594, 2.9321, 2.0594]

        joint_bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            joint_bounds.setLow(i, lower_limits[i])
            joint_bounds.setHigh(i, upper_limits[i])

        self.setBounds(joint_bounds)
        self.setup()


class MyStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, space_information):
        super(MyStateValidityChecker, self).__init__(space_information)
        self.space_information = space_information

    def isValid(self, state):
        return self.space_information.satisfiesBounds(state) and self.clearance(state) < 0.005
    
    def clearance(self, state):
        # q = ur_kin.random_joint_angles()
        q = []
        for i in range(6):
            q.append(state[i])
        q.append(0.00)
        p_trocar = np.array([[-0.63218696], [0.0000982], [0.760526907]])
        p_i = ur_kin.forward(q)[:3, 3]
        p_t = ee_kin.forward(q)[:3, 3]

        AP = p_trocar - p_i
        
        d = p_t - p_i
        

        lam = np.dot(np.transpose(AP), d)[0,0]/LA.norm(d)**2
        p_rcm = p_i + lam*(p_t - p_i)

        return np.linalg.norm(p_rcm - p_trocar)
    


class MyRRT:
    def __init__(self, ndof, step_size=0.05):
        self.ndof = ndof
        
        self.state_space = MyStateSpace(ndof)
        # self.control_space = MyControlSpace(self.state_space, ndof)
        self.simple_setup = og.SimpleSetup(self.state_space)
        si = self.simple_setup.getSpaceInformation()
        # si.setPropagationStepSize(step_size)
        # si.setMinMaxControlDuration(1, 1)
        # si.setDirectedControlSamplerAllocator(oc.DirectedControlSamplerAllocator(directedControlSamplerAllocator))

        # propagator = MyStatePropagator(self.simple_setup.getSpaceInformation(), ndof)
        # self.simple_setup.setStatePropagator(propagator)

        

        # self.planner = oc.KPIECE1(self.simple_setup.getSpaceInformation())
        # self.planner.setup()
        # ========= RRTConnect planner ============
        vc = MyStateValidityChecker(self.simple_setup.getSpaceInformation())
        self.simple_setup.setStateValidityChecker(vc)

        self.planner = og.RRTConnect(self.simple_setup.getSpaceInformation())
        p_goal = 0.05
        # self.planner.setGoalBias(p_goal)
        self.planner.setRange(0.05)


        self.simple_setup.setPlanner(self.planner)
        self.simple_setup.setup()
        
        


    def psolve(self, start, goal, timeout):
        self.simple_setup.setStartState(start)
        # mygoalregion = MyGoalRegion(self.simple_setup.getSpaceInformation(), goal, self.ndof)
        self.simple_setup.setGoalState(goal)

        
        
        # print(self.simple_setup)
        # print("Setup Done")
        

        if self.simple_setup.solve(timeout):
            print("Solved")
            if self.simple_setup.haveExactSolutionPath():
                print ("Exact Solution.")
                return self.simple_setup.getSolutionPath()
            elif self.simple_setup.haveSolutionPath():
                print ("Approximate Solution.")
                return self.simple_setup.getSolutionPath()
        else:
            print ("No Solution Found.")
            return None
        


if __name__ == '__main__':

    ndof = 6

    planner = MyRRT(ndof)

    start = ob.State(planner.state_space)
    goal = ob.State(planner.state_space)



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

    print(path)

