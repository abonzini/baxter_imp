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
from skimage import measure

origin = [38.6, 70.05, 7.5] # "Origin" of tripod base measured by hand using rviz on orange piece center in cms.
origin_angle = 45+13.75 # 45 degree is angle of arm and 13.75 the place where tripod is
h = 0 # Height of exploration cylinder
r = 0 # Radius of exploration cylinder
empty_point = -0.1
manual = False
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
        adjusted_angle = angle_xy * 180 / math.pi - origin_angle
        if adjusted_angle > 180:
            adjusted_angle -= 360
        elif adjusted_angle < -180:
            adjusted_angle += 360
        xy_mag = np.linalg.norm(direction[0:2])
        angle_z = np.arctan2(direction[2],xy_mag).ravel()[0]
        
        #print("xy angle",angle_xy,"angle_z",angle_z,"xy_mag",xy_mag,"adjusted_angle",adjusted_angle)
        rot_x = 0
        rot_y = 0
        rot_z = 0
        if (170 > adjusted_angle > 0) == (self.intuitive): # Inner part of robot... But angles closer to 180 are better with the other arm...
            print("Inner robot pose")
            rot_y += -math.pi/2+angle_z
            rot_z = angle_xy - math.pi
        else: # Outer part of robot...
            print("Outer robot pose")
            rot_y += math.pi/2 - angle_z
            rot_z = angle_xy

        orient = ts.quaternion_from_euler(rot_x, rot_y, rot_z) # Specific transform
        print("Target orientation of sensor is",orient,"for an xy angle",angle_xy,"and angle_z of",angle_z)
        return orient

    def CalculateEnterExitDistances(self, point, direction):
        safety_distance = 2.5 # Real expl will start and end 2.5cm outside the box for more margin (and also bc collision)
        
        # From the point, heading towards direction, there is an entry point from the sphere and an exit point, given by the formula of sphere-line intersection
        if(np.linalg.norm(direction[0:2])==0.0):
            enter_distance = float("-inf")
            exit_distance = float("inf")
        else:
            enter_distance, exit_distance = self.LineSphereIntersection(point[0:2], direction[0:2], r+safety_distance) # Now a circle only on plane xy
        
        if direction[2] != 0: # Check if the trajectory leaves the top of cilinders first
            upper_lid_distance = (h+safety_distance-point[2])/abs(direction[2])
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
            orient = self.CalculateUprightInnerOrient(direction)
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
        
        response = self.CallPlanner("pose_planner", req) # Call planner service
        if response is None or not response.success:
            return False

        total_traj.trajectory = [response.planned_trajectory]
        controls = []
        if interm_point is not None: # Then need to plan to interm point
            req.initial_position = req.destination_position # Load interm values
            req.initial_orientation = req.destination_orientation
            req.destination_position = list(interm_point)
            req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions

            response = self.CallPlanner("line_trajectory_planner", req) # Call planner service
            if response is None or not response.success:
                return False
            
            total_traj.trajectory += [response.planned_trajectory]
            controls += [COMMANDS.MEDIUM_MOVEMENT]
            self.last_start_pos_interm = req.destination_position

        # If initial planning is a success, will plan for the line trajectory
        req.initial_position = req.destination_position # Load interm values
        req.initial_orientation = req.destination_orientation
        req.destination_position = list(end_point)
        req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions

        response = self.CallPlanner("line_trajectory_planner", req) # Call planner service
        if response is None or not response.success:
            return False

        total_traj.trajectory += [response.planned_trajectory]
        controls += [COMMANDS.SLOW_MOVEMENT]
        
        #print("Planning successful")
        total_traj.arm = arm
        total_traj.control = [COMMANDS.MEDIUM_MOVEMENT_ADAPTED, COMMANDS.CALIBRATE_SENSOR, COMMANDS.ACTIVATE_SENSOR]+controls+[COMMANDS.DEACTIVATE_SENSOR]
        total_traj.side = side
        self.calculated_traj = total_traj
        return True
    
    def PlanNeutral(self, arm):
        req = PlannerServiceRequest()
        response = self.CallPlanner("neutral_planner", req) # Call planner service
        if response is None or not response.success:
            return None
        total_traj = ArmTrajectory()
        total_traj.arm = arm
        total_traj.side = "tip"
        total_traj.trajectory = [response.planned_trajectory]
        total_traj.control += [COMMANDS.MEDIUM_MOVEMENT_ADAPTED]
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
            
            #if self.last_start_pos_interm is not None:
                #req.destination_position = self.last_start_pos_interm
                #req.destination_orientation = self.last_start_orient
                
                #response = self.CallPlanner("line_trajectory_planner", req) # Call planner service
                #if response is None or not response.success:
                    #continue

                #req.initial_position = self.last_start_pos_interm
                #req.initial_orientation = self.last_start_orient
                #req.initial_joints = response.planned_trajectory.joint_trajectory.points[-1].positions
                #total_traj.trajectory += [response.planned_trajectory]
                #total_traj.control += [COMMANDS.FAST_MOVEMENT]
                
            req.destination_position = self.last_start_pos
            req.destination_orientation = self.last_start_orient

            response = self.CallPlanner("line_trajectory_planner", req) # Call planner service
            if response is None or not response.success:
                continue
            
            total_traj.trajectory += [response.planned_trajectory]
            total_traj.control += [COMMANDS.FAST_MOVEMENT]
            
            self.calculated_traj = total_traj
            break
        return True

    def MeasureContactPoint(self):
        rospy.sleep(1)
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        dest = self.last_start_arm[0] + "_gripper_" + self.last_start_arm[0] + "_finger_" + self.last_start_side
        tStamp = tfBuffer.lookup_transform("base", dest, rospy.Time(0),rospy.Duration(10.0)) # I got current point of robot where I want it
        point = [tStamp.transform.translation.x * 100, tStamp.transform.translation.y* 100, tStamp.transform.translation.z * 100] # From m to cm
        orient = [tStamp.transform.rotation.x, tStamp.transform.rotation.y, tStamp.transform.rotation.z, tStamp.transform.rotation.w]
        print("\tMeasured following tf, point:", point, "Orient:", orient)
        
        # Removed this part, not important
        '''
        sensor_name = "/"+dest+"_sensor_displacement"
        #print("Will check message:", sensor_name)
        sensor_disp = rospy.wait_for_message(sensor_name, SensorDisp)
        print("\tSensor displacement is",sensor_disp)
        trans = [sensor_disp.x_disp, sensor_disp.y_disp, 0.0]
        rot = [0, 0, 0, 1]
        trans_matrix = ts.concatenate_matrices(ts.translation_matrix(trans), ts.quaternion_matrix(rot))
        point_matrix = ts.concatenate_matrices(ts.translation_matrix(point), ts.quaternion_matrix(orient))
        result = np.dot(point_matrix, trans_matrix)
        res_pos = ts.translation_from_matrix(result)
        
        print("\tMeasured point",point,"And sensor displacement",trans,"For a resulting point of",res_pos)
        '''
        point = np.array(point)
        point -= origin
        
        #print("\tMeasured point",point)
        return point

    def GetBestPlan(self, point, direction):
        self.intuitive = True # Plan logically
        finger_length = 13
        #min_upwards_height = 3 # Fingertip can only move upwards if desired point is more than this height
        xy_angle = np.arctan2(direction[1],direction[0]).ravel()[0] * 180 / math.pi # Calculate the rotation angle on the xy plane, the fingertip will point here
        print("Absolute XY angle...",xy_angle)
        xy_angle -= origin_angle
        if xy_angle > 180:
            xy_angle -= 360
        elif xy_angle < -180:
            xy_angle += 360
        xy_mag = np.linalg.norm(direction[0:2])
        angle_z = np.arctan2(direction[2],xy_mag).ravel()[0] * 180 / math.pi
        print("xy angle",xy_angle,"angle_z",angle_z,"xy_mag",xy_mag)

        possible_plans = [] # Tuple with part direction and whether it is the intuitive sie of the robot (inner vs outer)
        if angle_z > 0: # If movement is trending upwards...
            direction[2] = 0 # will touch from side
            direction = direction / np.linalg.norm(direction)
            print("Will come from side")
        if angle_z < -45: # If planned comng from top at 45 degree, I will touch with tip
            print("Z angle is downwards", angle_z, "so tip is preferred")
            possible_plans += [("tip",np.array([0,0,-1]),True)] # First option coming from above
            possible_plans += [("tip",direction,True)] # Or then maybe coming with the real direction
        #elif (h - point[2]) < finger_length: # I'll do upright if I can
            #print("Will use upright finger orientation since h is",h, "and desired point at",point[2])
            #direction[2] = 0 # will touch from side
            #direction = direction / np.linalg.norm(direction)
            #interm_point  = point - (direction * 2) # Move 2 cm to the back
            #interm_direction = np.array([0,0,-1]) # Coming from up
            #plan_success = self.Plan(point,direction,"left","inner_vert", interm_point = interm_point, interm_direction = interm_direction)
        elif 90 >= abs(xy_angle) >= 20: # This area is better to touch with tip
            print("XY angle is", xy_angle, "so tip is preferred before side")
            possible_plans += [("tip",direction,True)]
            possible_plans += [("inner",direction,True)]
        else: # Other area better with inner
            print("Side preferred")
            possible_plans += [("inner",direction,True)] # Will try everything
            possible_plans += [("inner",direction,False)]
            if 110 >= abs(xy_angle):
                possible_plans += [("tip",direction,True)]
        
        for possible_plan in possible_plans:
            self.intuitive = possible_plan[2]
            plan_success = self.Plan(point,possible_plan[1],"left",possible_plan[0])
            if plan_success: break
        self.last_calculated_direction = possible_plan[1]
        # Return all plans in the prioritized order and the new, modified direction
        return plan_success

    def GoToPoint(self, point, direction, max_times = None):
        movement_completed = False
        contact_pos = None
        contact_dir = None
        # Initial movement loop
        while not movement_completed:
            if max_times is not None: # If there's a number of max times
                max_times -= 1
                if max_times < 0:
                    print("MAX NUMBER OF TRIES REACHED, ABORTING THIS POINT")
                    return contact_pos, contact_dir, movement_completed
            # Planning stage loop
            plan_success = False
            plan_success = self.GetBestPlan(point, direction)
            if not plan_success:
                input("Planning failing...")
            #Move loop
            if plan_success:
                movement = self.Move(self.calculated_traj)
                if movement is None or not movement.finished_succesfully:
                    show_info("Something failed during movement, I will try again")
                else:
                    movement_completed = True
        if movement.touch_detected:
            contact_pos = self.MeasureContactPoint()
            contact_dir = self.last_calculated_direction
        # Finally a try of going out
        self.PlanOut()
        self.Move(self.calculated_traj)
        # Return obtained contact points
        return contact_pos, contact_dir, movement_completed

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

def Plot_Mesh(verts, faces, vert_covs, r, h, max_cov_color = 0.2, colormap_color = 'coolwarm'):
    face_uncertainty = vert_covs[faces] # Now each polygon contains 3 bordering uncertainties
    face_uncertainty = np.amax(face_uncertainty, axis = 1)
    face_uncertainty = np.log(face_uncertainty).ravel() # Will do logarithm, easier to visualize
    norm = plt.Normalize(vmin=math.log(0.05**2), vmax=math.log(max_cov_color**2)) # Will normalize, min covar is 0, max is 1
    colormap = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap_color)) # Create a cmap class
    face_uncertainty = colormap.to_rgba(face_uncertainty) # Transform cov matrix into RGB matrix colored, 1 is warm, 0 is cool

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces]) # Create a number of polygons made by combinations of vertexes
    mesh.set_facecolor(face_uncertainty)
    mesh.set_edgecolor('k')
    mesh.set_linewidth(0.1)
    ax.clear()
    r += 0.5
    h += 1
    ax.set_title("STEP {0}".format(step))
    ax.set_xlim(-h/2,h/2)  # Set x, y, z limits
    ax.set_ylim(-h/2,h/2)
    ax.set_zlim(0, h)
    ax.azim = 45
    ax.add_collection3d(mesh) # Add collection to plot
    plt.show(block=False)

def main():
    rospy.init_node('ExpNoSymNode')
    global show_info
    if manual:
        show_info = input
    else:
        show_info = print
    global Planner
    Planner = ContactPlanner()
    
    #Create folder to save the function's exploration
    foldername = "./exploration_auto_implicit"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    n_points = 0
    subfoldername = foldername + '/try' #Create a folder to save this experiment's data
    i = 0
    while os.path.exists(subfoldername + "_" + str(i)):
        i += 1
    subfoldername = subfoldername + "_" + str(i)
    os.makedirs(subfoldername)
    
    n_points_max = 40 # Max amount of points I will measure (before we decide some criterion for stopping exploration)
    #Config R as the maximum radius of expl area centered in origin (usually object height?)
    global r
    global h
    #mustard
    #h = 20.0 #IN CM
    #r = 6.5
    #Chips
    h = 25
    r = 6
    #SmallBot
    #h = 18.0 #IN CM
    #r = 5
    #Windex
    #h = 30.0 #IN CM
    #r = 7.0
    #Jar
    #h = 26.0 #IN CM
    #r = 10.0
    #Pear
    #h = 12.0 #IN CM
    #r = 4.5
    # Cup
    #h = 16.0 #IN CM
    #r = 12.0
    
    # Send bounds to motion planner to move around sphere avoiding collisions
    bounds = SetBoundsMsg()
    bounds.sphere_center = list(np.array(origin)/100)
    bounds.radius = 0.707*r/100
    bounds.height = (h+2)/100
    pub = rospy.Publisher('exploration_bounds', SetBoundsMsg, queue_size=10)
    print("Creating ros publishers")
    rospy.sleep(2)
    pub.publish(bounds)

    bound_R = math.sqrt((h/2)**2 + r**2)+1 # From center of exploration area, max dist is lid of cilinder, and 1 bc may measure 1cm outside
    prior_calc, prior_deriv_calc = SpherePriorCalculator(2*r/3,2*r/3,2*h/3, min(r,h)/2) # Assume prior ellipsoid of 2/3 r and h, h half value i guess?
    
    ImpSurf = GP()
    ImpSurf.hyp = [2*bound_R]
    ImpSurf.noise = 0.05 # Lower noise in implicits

    # Test points where we will sample our surface model
    resolution = 17 # Resolution should be low enough to not take a lot of time, but will allow good exploration!
    bound_range_x = np.linspace(-r, r, resolution)
    bound_range_y = np.linspace(-r, r, resolution)
    bound_range_z = np.linspace(0, h, resolution)
    spacing = [np.diff(bound_range_x)[0],np.diff(bound_range_y)[0],np.diff(bound_range_z)[0]]
    xx, xy, xz = np.meshgrid(bound_range_x,bound_range_y,bound_range_z, indexing='ij')
    mesh_shape = xx.shape
    Xx = np.hstack((xx.reshape(-1,1),xy.reshape(-1,1),xz.reshape(-1,1))) # Xx contains all points
    xx_shape = xx.shape
    del xx, xy, xz, bound_range_x, bound_range_y, bound_range_z # Delete these guys since they take a lot of space!

    ImpSurf.AddXx(Xx)
    prior_data = prior_calc(Xx)
    
    #Planner.GoToPoint(np.array([0,0,0]), np.array([0,0,-1])) # Use to check if centered

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')    
    while(n_points<n_points_max): # Begin exploration
        mu = np.copy(ImpSurf.mu)
        mu += prior_data
        mu = mu.reshape(xx_shape) # Organize space estimation on the corresponding axes
        # Marching cubes estimation!
        verts, faces, normals, _ = measure.marching_cubes(mu, 0) # Will get the obj estimation of marching cubes
        verts *= spacing # Multiply by original "scale"
        verts -= np.array([r,r,0]) # Now verts has actual, centered points

        # Calculate uncertainty of each vertex point
        vertGP = ImpSurf.GetCopy() # Copy everything except Xx from GP (data)
        vertGP.AddXx(verts) # Add mesh vertices to gp estimation
        vertCov = np.diag(vertGP.cov).reshape((-1,1)) # I obtain a list of covs for each vertex
        del vertGP # To free some space...

        Plot_Mesh(verts, faces, vertCov, r, h) # Plot (quickly) and continue my work
        plot_ok = input("Do you like what you see? N/n (discards previous point)")
        if plot_ok == 'n' or plot_ok == 'N':
            ImpSurf.RemoveX(1) # Remove last point
            n_points -= 1 # Return to old point
            continue

        print("Surface estimation obtained, will go to face with most uncertainty")
        vert_order = np.argsort(vertCov, axis=0) # Sort by uncertainty
        if n_points == 0:
            print("This is the first point so I will explore a random one!")
            vert_order = np.random.shuffle(vert_order)
        finished = False
        next_candidate = vertCov.shape[0] # Start from last
        while not finished:
            next_candidate -= 1
            if next_candidate<0:
                print("Fatal error?! No faces could be explored")
                return
            n = vert_order[next_candidate] # Need to find the once with HIGHEST uncertainty
            print("New candidate is",verts[n].ravel())
            print("Estimated Approach Gradient is",-normals[n].ravel())
            input("If reached here, baxter will try to move")
            touch_location, gradient, finished = Planner.GoToPoint(verts[n].ravel(), -normals[n].ravel(), 5) # We try 5 times
            if not finished or touch_location is None:
                print("Couldn't explore this face succesfully, will try the next")

        newX = touch_location.reshape((1,-1))
        print("Touched point", newX)
        newY = -prior_calc(newX)
        ImpSurf.Y = np.vstack((ImpSurf.Y, newY))
        ImpSurf.AddX(newX)

        print("N points is",n_points)
        n_points += 1
        filename = subfoldername + '/' + str(n_points)
        with open(filename+'.gp', 'wb') as f:
            pickle.dump(ImpSurf, f)

if __name__ == "__main__":
  main()
