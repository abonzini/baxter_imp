import numpy as np
import os
import scipy.optimize as opt
#!/usr/bin/env python
import random
import itertools
import time
import pickle
import math

import sys
import rospy
import tf
from tf import transformations as ts
import tf2_ros
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
import actionlib
from baxter_imp import *
from baxter_imp.msg import ArmTrajectory, MoveAndContactFeedback, MoveAndContactResult, MoveAndContactAction
from baxter_imp.msg import *
from baxter_imp.srv import *

origin = [66.5, 0,  10.8] # "Origin" of tripod base measured by hand using rviz on orange piece center in cms. Using left hand since for some reason is 1cm shorter (and I dont want negative numbers
h = 0 # Height of exploration cylinder
r = 0 # Radius of exploration cylinder
manual = True
show_info = None

Planner = None

class COMMANDS:
    SLOW_MOVEMENT = 0
    MEDIUM_MOVEMENT = 1
    FAST_MOVEMENT = 2
    MEDIUM_MOVEMENT_ADAPTED = 3
    CALIBRATE_SENSOR = 4
    ACTIVATE_SENSOR = 5
    DEACTIVATE_SENSOR = 6

class ContactPlanner(object):
    last_start_pos = None
    last_interm_pos = None
    last_orient = None
    last_start_side = None
    last_start_arm = None

    last_calculated_direction = None
    calculated_traj = None
    
    left_crossed = False
    right_crossed = False
    
    def LineSphereIntersection(self, line_point, line_dir, sphere_radius): # Will return 2 distances, one to exit sphere and another to enter sphere
        #Sphere assumed with origin on 0,0,0
        # Quadratic equation stolen from wikipedia
        dot = np.dot(line_dir, line_point)
        mod = np.linalg.norm(line_dir)**2
        root = (2*dot)**2 - 4 * mod * (np.linalg.norm(line_point)**2 - sphere_radius**2)
        root = math.sqrt(root)
        exit_distance = - 2 * dot + root
        enter_distance = - 2 * dot - root
        exit_distance /= 2 * mod
        enter_distance /= 2 * mod
        return enter_distance, exit_distance
    
    def CalculateTipOrient(self, direction, arm): # Given a frame, I calculate the shortest rotation from frame to my vector
        ref_frame = np.array([0,0,1])
        #Crafty way of obtaining a quaternion from [0,0,1] to any desired vector
        if(np.dot(direction,ref_frame)==1):
            return np.array([0,0,0,1])
        if(np.dot(direction,ref_frame)==-1):
            return np.array([1,0,0,0]) # Rotate over.. x axis (random choice
        mid_vector = (direction + ref_frame)/2 # Calculate vector located between vectors (its angle will be theta/2 from ref)
        sin_axis = np.cross(ref_frame, mid_vector) # The axis is orthogonal to both vectors, and its module is sin(theta/2)*axis, perfect for quaternion
        cos = np.dot(ref_frame, mid_vector) # cos(theta/2)
        orient = sin_axis.ravel()
        orient = np.append(orient, [cos]) # Create quaternion that expresses my new orientation
        return orient

    def CalculateUprightInnerOrient(self, direction):
        angle = np.arctan2(direction[1],direction[0]).ravel()[0] # Calculate the rotation angle on the xy plane, the fingertip will point here
        orient = ts.quaternion_from_euler(math.pi/2, math.pi, (math.pi/2)+angle) # Specific transform to transform from world coordinates to fingertip pointing to direction
        return orient

    def CalculateSidewaysInnerOrient(self, direction, arm):
        angle_xy = np.arctan2(direction[1],direction[0]).ravel()[0] # Calculate the rotation angle on the xy plane, the fingertip will point here
        xy_mag = np.linalg.norm(direction[0:2])
        angle_z = np.arctan2(direction[2],xy_mag).ravel()[0]
        
        rot_x = 0
        if arm == "left":
            rot_x = math.pi
        rot_y = 0
        rot_z = 0
        if arm == "right":
            rot_y = math.pi
        if direction[0] > 0: # Means it is coming from the back...
            rot_y += -math.pi/2-angle_z
            rot_z = angle_xy
        else:
            rot_y += math.pi/2 + angle_z
            rot_z = angle_xy-math.pi

        orient = ts.quaternion_from_euler(rot_x, rot_y, rot_z) # Specific transform
        return orient

    def CalculateEnterExitDistances(self, point, direction):
        # From the point, heading towards direction, there is an entry point from the sphere and an exit point, given by the formula of sphere-line intersection
        if(np.linalg.norm(direction[0:2])==0.0):
            enter_distance = float("-inf")
            exit_distance = float("inf")
        else:
            enter_distance, exit_distance = self.LineSphereIntersection(point[0:2], direction[0:2], r) # Now a circle only on plane xy
        
        if direction[2] != 0: # Check if the trajetory leaves the top of cilinders first
            upper_lid_distance = (h-point[2])/abs(direction[2])
            lower_lid_distance = (point[2])/abs(direction[2])
            if direction[2] > 0: # If going up...
                enter_distance2 = -lower_lid_distance
                exit_distance2 = upper_lid_distance
            else:
                enter_distance2 = -upper_lid_distance
                exit_distance2 = lower_lid_distance
            enter_distance = max(enter_distance,enter_distance2) # Choose soonest entrance
            exit_distance = min(exit_distance, exit_distance2) # Choose the soonest exit
        return enter_distance, exit_distance
    
    def CallPlanner(self, name, req):
        rospy.wait_for_service(name)
        try:
            motion_planner = rospy.ServiceProxy(name, PlannerService)
            response = motion_planner(req)
            return response
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            return None

    def Plan(self, point, direction, arm, side, interm_point = None, interm_direction = None):
        enter_distance, exit_distance = self.CalculateEnterExitDistances(point, direction)
        if interm_point is not None: # If will do it in 2 parts, will override start point
            enter_distance,_ = self.CalculateEnterExitDistances(interm_point, interm_direction)
        #print("Enter dist:", enter_distance, "Exit:", exit_distance)

        safety_distance = 5 # Real expl will start and end 5cm outside the box for more margin (and also bc collision)
        enter_distance -= safety_distance
        # Finally calculate the start and end point of this exploration
        if interm_point is None:
            start_point = point + (enter_distance*direction)
        else:
            start_point = interm_point + (enter_distance*interm_direction)
        end_point = point + (exit_distance*direction)
        
        show_info("Trajectory starts at "+str(start_point)+" and ends at " +str(end_point))
        show_info("About to plan")
        #Transform to robot and meters
        start_point += origin
        start_point /= 100
        end_point += origin
        end_point /= 100
        if interm_point is not None:
            interm_point += origin
            interm_point /= 100
            
        total_traj = ArmTrajectory()
        if side == "inner":
            orient = self.CalculateSidewaysInnerOrient(direction,arm)
        elif side == "inner_vert":
            orient = self.CalculateUprightInnerOrient(direction,arm)
            side = "inner"
        else:
            orient = self.CalculateTipOrient(direction,arm)
        #print("With orientation",orient)
        
        req = PlannerServiceRequest()
        req.destination_position = list(start_point)
        req.destination_orientation = list(orient)
        req.sensor_side = side
        #Save for way back
        self.last_start_pos = req.destination_position
        self.last_start_orient = req.destination_orientation
        self.last_start_side = side
        self.last_start_arm = arm
        self.last_start_pos_interm = None
        
        response = self.CallPlanner(arm + "_group/pose_planner", req) # Call planner service
        if response is None or not response.success:
            return False

        total_traj.trajectory = [response.planned_trajectory]
        controls = []
        if interm_point is not None: # Then need to plan to interm point
            req.initial_position = req.destination_position # Load interm values
            req.initial_orientation = req.destination_orientation
            req.destination_position = list(interm_point)
            req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions

            response = self.CallPlanner(arm + "_group/line_trajectory_planner", req) # Call planner service
            if response is None or not response.success:
                return False
            
            total_traj.trajectory += [response.planned_trajectory]
            controls += [COMMANDS.SLOW_MOVEMENT]
            self.last_start_pos_interm = req.destination_position

        # If initial planning is a success, will plan for the line trajectory
        req.initial_position = req.destination_position # Load interm values
        req.initial_orientation = req.destination_orientation
        req.destination_position = list(end_point)
        req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions

        response = self.CallPlanner(arm + "_group/line_trajectory_planner", req) # Call planner service
            if response is None or not response.success:
                return False

        total_traj.trajectory += [response.planned_trajectory]
        controls += [CONTROLS.SLOW_MOVEMENT]
        
        #print("Planning successful")
        total_traj.arm = arm
        total_traj.control = [CONTROLS.MEDIUM_MOVEMENT_ADAPTED, CONTROLS.CALIBRATE_SENSOR, CONTROLS.ACTIVATE_SENSOR]+controls+[CONTROLS.DEACTIVATE_SENSOR]
        total_traj.side = side
        self.calculated_traj = total_traj
        return True
    
    def PlanNeutral(self, arm):
        req = PlannerServiceRequest()
        response = self.CallPlanner(arm + "_group/neutral_planner", req) # Call planner service
        if response is None or not response.success:
            continue
        else:
            break
        total_traj = ArmTrajectory()
        total_traj.arm = arm
        total_traj.side = "tip"
        total_traj.trajectory = [response.planned_trajectory]
        total_traj.control += [CONTROLS.MEDIUM_MOVEMENT]
        return total_traj
    
    def PlanOut(self):
        while True:
            req = PlannerServiceRequest()
            total_traj = ArmTrajectory()
            total_traj.arm = self.last_start_arm
            total_traj.side = self.last_start_side
            total_traj.trajectory = []
            total_traj.control = []
            req.sensor_side = self.last_start_side
            
            if self.last_start_pos_interm is not None:
                req.destination_position = self.last_start_pos_interm
                req.destination_orientation = self.last_start_orient
                
                response = self.CallPlanner(self.last_start_arm + "_group/line_trajectory_planner") # Call planner service
                if response is None or not response.success:
                    continue

                req.initial_position = self.last_start_pos_interm
                req.initial_orientation = self.last_start_orient
                req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions
                total_traj.trajectory += [response.planned_trajectory]
                control += [CONTROLS.FAST_MOVEMENT]
                
            req.destination_position = self.last_start_pos
            req.destination_orientation = self.last_start_orient

            response = self.CallPlanner(self.last_start_arm + "_group/line_trajectory_planner") # Call planner service
            if response is None or not response.success:
                continue
            
            total_traj.trajectory += [response.planned_trajectory]
            total_traj.control += [CONTROLS.FAST_MOVEMENT]
            
            self.calculated_traj = total_traj
            break
        return True

    def CheckIfArmCrosses(self, arm, pos):
        pos = pos.ravel()
        if arm == "left" and pos[1] < 0:
            return True
        if arm == "right" and pos[1] > 0:
            return True
        return False

    def MeasureContactPoint(self):
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        dest = self.last_start_arm[0] + "_gripper_" + self.last_start_arm[0] + "_finger_" + self.last_start_side
        tStamp = tfBuffer.lookup_transform("base", dest, rospy.Time(0),rospy.Duration(10.0)) # I got current point of robot where I want it
        point = [tStamp.transform.translation.x * 100, tStamp.transform.translation.y* 100, tStamp.transform.translation.z * 100] # From m to cm
        orient = [tStamp.transform.rotation.x, tStamp.transform.rotation.y, tStamp.transform.rotation.z, tStamp.transform.rotation.w]
        #print("Measured following tf, point:", point, "Orient:", orient)
        
        sensor_name = "/"+dest+"_sensor_displacement"
        #print("Will check message:", sensor_name)
        sensor_disp = rospy.wait_for_message(sensor_name, SensorDisp)
        #print("Sensor displacement is",sensor_disp)
        trans = [sensor_disp.x_disp, sensor_disp.y_disp, 0.0]
        rot = [0, 0, 0, 1]
        trans_matrix = ts.concatenate_matrices(ts.translation_matrix(trans), ts.quaternion_matrix(rot))
        point_matrix = ts.concatenate_matrices(ts.translation_matrix(point), ts.quaternion_matrix(orient))
        result = np.dot(point_matrix, trans_matrix)
        res_pos = ts.translation_from_matrix(result)
        #show_info("Total movement is then:"+ str(res_pos))
        
        res_pos = np.array(res_pos)
        res_pos -= origin

        if self.last_start_arm == "left": # Will check if arm is in a "dangerous" place
            left_crossed = self.CheckIfArmCrosses("left", res_pos)
        else:
            right_crossed = self.CheckIfArmCrosses("right", res_pos)

        return res_pos

    def PlanningPriorityDecisions(self, point, direction):
        finger_length = 13
        #min_upwards_height = 3 # Fingertip can only move upwards if desired point is more than this height
        xy_angle = np.arctan2(direction[1],direction[0]).ravel()[0] * 180 / math.pi # Calculate the rotation angle on the xy plane, the fingertip will point here
        xy_mag = np.linalg.norm(direction[0:2])
        angle_z = np.arctan2(direction[2],xy_mag).ravel()[0] * 180 / math.pi

        plans = {
            "left": [],
            "right": []
        }
        def GetPrefferedArm(left):
            if left:
                return "left"
            else:
                return "right"
        left_arm_pref = True # By default I try to use left arm, since it works better
        if direction[1] > 0:
            left_arm_pref = False # If the mov is from right to left then its better to do with right arm
        
        if angle_z > 0: # If movement is trending upwards...
            direction[2] = 0 # will touch from side
            direction = direction / np.linalg.norm(direction)
            print("Will come from side")
        if 100 >= abs(xy_angle) >= 45: # This area is better to touch with tip
            print("XY angle is", xy_angle, "so tip is preferred before side")
            plans[GetPrefferedArm(left_arm_pref)] = ["tip","inner"]
        else: # Other area better with inner
            if xy_mag > 0.0: # If planned center point, I will touch with tip
                print("XY angle is", xy_angle, "so inner is preferred")
                plans[GetPrefferedArm(left_arm_pref)] = ["inner", "tip"]
                plans[GetPrefferedArm(not left_arm_pref)] = ["inner"] # Maybe the other arm can plan with inner anyway 
            else:
                plans[GetPrefferedArm(left_arm_pref)] = ["tip"] # Otherwise only tip to touch "top" of object
                plans[GetPrefferedArm(not left_arm_pref)] = ["tip"]
            if 0<abs(xy_angle)<45: # But on the extreme region is better to actually switch arm
                print("Tricky area, better try with the other arm and coming from side")
                left_arm_pref = not left_arm_pref # toggle
                direction[2] = 0 # will touch from side, no angles
                direction = direction / np.linalg.norm(direction)

        self.last_calculated_direction = direction
        # Return all plans in the prioritized order and the new, modified direction
        return [(GetPrefferedArm(left_arm_pref), plans[GetPrefferedArm(left_arm_pref)]),(GetPrefferedArm(not left_arm_pref), plans[GetPrefferedArm(not left_arm_pref)])]

    def CalculateBestPlan(self, point, direction):
        plans_order = self.PlanningPriorityDecisions(point, direction)

        for arm_plan in plan_order:
            print("For arm", arm_plan[0])
            for side in arm_plan[1]:
                print("Planning touch with", side)
                plan_success = self.Plan(point,direction,arm,side)
                if plan_success:
                    return plan_success

    def GoToPoint(self, point, direction):
        movement_completed = False
        clean_slate = False
        # Initial movement loop
        while not movement_completed:
            # Planning stage loop
            plan_success = False
            while not plan_success: # Will try to plan until success
                plan_success = self.CalculateBestPlan(point, direction)
            # Uncross arm loop
            arm_to_move = None
            arm_will_cross = self.CheckIfArmCrosses(self.calculated_traj.arm, point) # Will check if trajectory may result in collision between arms
            if self.calculated_traj.arm == "right" and (arm_will_cross or left_crossed) and not clean_slate:
                arm_to_move = "left"
            elif self.calculated_traj.arm == "left" and (arm_will_cross or right_crossed) and not clean_slate:
                arm_to_move = "right"
            while arm_to_move is not None:
                move_neutral_plan = self.PlanNeutral(arm_to_move) # Plan ot return
                movement = self.Move(move_neutral_plan)
                if movement is None or not movement.finished_succesfully:
                    input("Something failed while returning, I will try again")
                else:
                    arm_to_move = None
                    clean_slate = True # Won't repeat this...
            #Move loop
            movement = self.Move(self.calculated_traj)
            if movement is None or not movement.finished_succesfully:
                input("Something failed during movement, I will try again")
            else:
                movement_completed = True
        if not movement.touch_detected:
            contact_pos = None
            contact_dir = None
        else:
            contact_pos = self.MeasureContactPoint()
            contact_dir = self.last_calculated_direction
        # Finally a try of going out
        self.PlanOut()
        self.Move(self.calculated_traj)
        # Return obtained contact points
        return contact_pos, contact_dir

    def Move(self, traj):
        show_info("About to try movement")
        try:
            client = actionlib.SimpleActionClient('/' + traj.arm + '_group/movement_manager', MoveAndContactAction)
            client.wait_for_server()
            goal = MoveAndContactGoal()
            goal.traj = traj
            client.send_goal(goal)
            client.wait_for_result()
            return client.get_result()
        except rospy.ROSException as e:
            print("Action call failed: %s"%e)
            return None

def main():
    rospy.init_node('ExpNoSymNode')
    global show_info
    if manual:
        show_info = input
    else:
        show_info = print
    global Planner
    Planner = ContactPlanner()
    
    n_points_max = 40 # Max amount of points I will measure (before we decide some criterion for stopping exploration)
    noise_std = 0.05 # Lower noise in implicits
    #Config R as the maximum radius of expl area centered in origin (usually object height?)
    global r
    global h
    #mustard
    #h = 20.0 #IN CM
    #r = 6.5
    #Chips
    h = 25
    r = 5
    #SmallBot
    #h = 18.0 #IN CM
    #r = 5
    #Jar
    #h = 26.0 #IN CM
    #r = 12.0
    #Pear
    #h = 12.0 #IN CM
    #r = 4.5
    bound_R = math.sqrt((h/2)**2 + r**2)+1 # From center of exploration area, max dit is lid of cilinder, and 1 bc may measure 1cm outside
    hyp = [2*bound_R]
    prior_calc, prior_deriv_calc = SpherePriorCalculator(2*r/3,2*r/3,2*h/3, min(r,h)/2) # Assume prior ellipsoid of 2/3 r and h, h half value i guess?
    # Send bounds to motion planner to move around sphere avoiding collisions
    bounds = SetBoundsMsg()
    bounds.sphere_center = list(np.array(origin)/100)
    bounds.radius = r/100
    bounds.height = h/100
    pub1 = rospy.Publisher('/left_group/exploration_bounds', SetBoundsMsg, queue_size=10)
    pub2 = rospy.Publisher('/right_group/exploration_bounds', SetBoundsMsg, queue_size=10)
    print("Creating ros publishers")
    rospy.sleep(2)
    pub1.publish(bounds)
    pub2.publish(bounds)
    
    empty_point = -0.1 # If a point is registered as empty, will be a (small) negative value
    close_distance = 1 # we assume 1cm as a "close" distance, to add derivative information. Also to decide when assumed point and real contact are practically the same 
    far_distance = 7 # 7cm is assumed too much distance, to decide on the "far away touches"
    #Create folder to save the function's exploration
    foldername = "./exploration_auto_implicit"
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Exploration of initial points (a cilinder/ellipsoid is assumed)
    print("Will start by exploring some set points")
    # All data
    Y_priors = np.empty([0,1])
    Y = np.empty([0,1])
    X = np.empty([0,3])
    # Data of touches
    X_touch = np.empty([0,3])
    n_touch = 0
    # Data of empty spaces
    X_miss = np.empty([0,3])

    subdivisions = 4 # How many points I explore from top to bottom (top one is always at (0,0,top)
    heights = np.linspace(h, 0.0, num = subdivisions, endpoint = False).ravel()
    print("heights:",heights)
    for i in range(subdivisions): # For each vertical subdivision of sphere
        if heights[i] == h:
            new_target = np.array([0,0,h])
        else:
            angle = 2 * (i-1)* 2*math.pi/(subdivisions-1) # Will skip every one
            new_coord = [r,angle,heights[i]] # new point to explore on side of cylinder
            new_target = CylinderToCartesian(new_coord) # Transform into cartesian (robot)
        print("New candidate is",new_target)
        gradient = -new_target/np.linalg.norm(new_target)
        gradient = gradient.ravel()
        if np.linalg.norm(gradient[0:2]) > 0.0:
            gradient[2] = 0
            gradient[0:2] /= np.linalg.norm(gradient[0:2])
        print("Estimated Approach Gradient is", gradient)
        new_target = new_target.reshape((1,-1)) 
            
        touch_location, gradient = Planner.GoToPoint(new_target.ravel(), gradient)

        if touch_location is None:
            print("As no touch was detected, will just register empty point")
            X = np.vstack((X,new_target))
            X_miss = np.vstack((X_miss,new_target))
            Y = np.vstack((Y,[empty_point]))
            Y_priors = np.vstack((Y_priors,prior_calc(new_target)))
        else:
            print("Measured point",touch_location.ravel())
            outside_location = touch_location - gradient * close_distance
            Y = np.vstack(([0], Y, [empty_point]))
            X = np.vstack((touch_location, X, outside_location))
            Y_priors = np.vstack((prior_calc(touch_location.reshape((1,-1))),Y_priors,prior_calc(outside_location.reshape((1,-1)))))
            X_miss = np.vstack((X_miss,outside_location))
            X_touch = np.vstack((touch_location,X_touch))
            n_touch += 1

    '''        
    print("Also I add the point in the bottom") # Maybe no need, so that surface continues indefinitely unti prior
    touch_location = np.array([0,0,-0.5]).reshape(1,-1)
    outside_location = np.array([0,0,-1.5]).reshape(1,-1)
    Y = np.vstack(([0], Y, [empty_point]))
    X = np.vstack((touch_location, X, outside_location))
    Y_priors = np.vstack((prior_calc(touch_location),Y_priors,prior_calc(outside_location)))
    X_miss = np.vstack((X_miss,outside_location))
    X_touch = np.vstack((touch_location,X_touch))
    n_touch += 1'''

    K = Kernel(X,hyp) + noise_std**2 * np.eye(X.shape[0]) # Creation of GP model...
    L = np.linalg.cholesky(K)
    
    n_points = 0
    subfoldername = foldername + '/try' #Create a folder to save this experiment's data
    i = 0
    while os.path.exists(subfoldername + "_" + str(i)):
        i += 1
    subfoldername = subfoldername + "_" + str(i)
    os.makedirs(subfoldername)
    filename = subfoldername + '/' + str(n_points)
    np.savez(filename, K, L, X_touch, X_miss, Y, Y_priors) # I save invK, Xs and Y for future plotting and whatever analysis I want to make
    
    while(n_points<n_points_max):
        #Diff Evolution Optimization (Apparently the better-working)
        optimizer = GPOptHelper(X, Y-Y_priors, L, hyp, prior = prior_calc)

        funct_minimize = optimizer.OptNegCov
        funct_edge = optimizer.OptMu
        
        EdgeTol = 0.02
        Bounds = ([0, r], [0,2*math.pi], [1,h])
        nlc = opt.NonlinearConstraint(funct_edge, -EdgeTol, EdgeTol) # Only look for points in surface

        res = opt.differential_evolution(funct_minimize,Bounds,constraints=(nlc,),polish=False)
        if not res.success:
            print('OPT Failed at some point to find the next point to explore')
            break
        print()
        print("Next point has an uncertainty of",-res.fun)

        new_target = res.x
        gp = optimizer.GetGP(new_target.ravel())
        new_target = gp.Xx.reshape((1,-1))
        
        gradient = np.array([gp.Mu(model = TSPCov_ithDerivative(0)),gp.Mu(model = TSPCov_ithDerivative(1)),gp.Mu(model = TSPCov_ithDerivative(2))]).ravel()
        prior_grad = prior_deriv_calc(new_target).ravel()
        #print("Grad pre prior:", gradient, "prior grad...", prior_grad)
        gradient += prior_grad
        gradient /= np.linalg.norm(gradient)
        gradient = gradient.ravel()

        print("New candidate is",new_target)
        print("Estimated Approach Gradient is",gradient)
        
        touch_location, gradient = Planner.GoToPoint(new_target.ravel(), gradient)   
        
        new_X_touch = np.empty([0,3])
        new_X_miss = np.empty([0,3])
        new_Y_touch = np.empty([0,1])
        new_Y_miss = np.empty([0,1])
        if touch_location is None:
            print("As no touch was detected, will just register empty point")
            new_X_miss = np.vstack((new_X_miss,new_target))
            new_Y_miss = np.vstack((new_Y_miss,[empty_point]))
        else:
            distance = np.linalg.norm(new_target-touch_location) # How good was the estimation vs real touch?
            print("Touched at", touch_location, "at a distance of", distance,"of expected location")
            if distance < close_distance: # Means we have practically the expected point...
                pass
            elif distance > far_distance: # Point is too far from expected
                d = round(distance/far_distance)
                d_step = distance/d
                for i in range(int(d)):
                    dist = d_step * (i+1)
                    empty_measurement = new_target - gradient * dist
                    new_X_miss = np.vstack((new_X_miss,empty_measurement))
                    new_Y_miss = np.vstack((new_Y_miss, [empty_point]))
                print("Saved",d, "midpoints at a separation of", d_step)
            else: # Normal point
                empty_measurement = touch_location - gradient * close_distance
                new_X_miss = np.vstack((new_X_miss,empty_measurement))
                new_Y_miss = np.vstack((new_Y_miss, [empty_point]))
            new_X_touch = np.vstack((touch_location, new_X_touch))
            n_touch += 1
            new_Y_touch = np.vstack((np.array([0]), new_Y_touch))
            print("New X are:", new_X_touch, new_X_miss, "with values", new_Y_touch, new_Y_miss)
        
        filename = subfoldername + '/' + str(n_points)
        # Rest of GP
        # Add touches 
        if new_X_touch.any():
            Kx = Kernel(X,hyp,X2=new_X_touch)
            Kxx = Kernel(new_X_touch,hyp) + noise_std**2 * np.eye(new_X_touch.shape[0])
            K = np.block([[Kxx, Kx.T], [Kx, K]]) # Add new elements regarding new X to kernel K
            X = np.vstack((new_X_touch, X))
            X_touch = np.vstack((new_X_touch, X_touch))
        if new_X_miss.any():
            Kx = Kernel(X, hyp, X2= new_X_miss)
            Kxx = Kernel(new_X_miss,hyp) + noise_std**2 * np.eye(new_X_miss.shape[0])
            K = np.block([[K, Kx],[Kx.T, Kxx]])
            X = np.vstack((X, new_X_miss))
            X_miss = np.vstack((X_miss, new_X_miss))
        
        Y = np.vstack((new_Y_touch, Y, new_Y_miss)) # New members of dataset!
        Y_priors = np.vstack((prior_calc(new_X_touch),Y_priors,prior_calc(new_X_miss)))
        n_points += 1

        L = np.linalg.cholesky(K)

        filename = subfoldername + '/' + str(n_points)
        np.savez(filename, K, L, new_X_touch, new_X_miss, Y, Y_priors)

if __name__ == "__main__":
  main()
