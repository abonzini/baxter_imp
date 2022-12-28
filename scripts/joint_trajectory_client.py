#!/usr/bin/env python

import argparse
import sys

from copy import copy
import rospy
import actionlib
import baxter_interface
import baxter_tools
import numpy as np

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandActionGoal
)
from std_msgs.msg import String
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
from sensor_msgs.msg import JointState
from baxter_imp.msg import ArmTrajectory, MoveAndContactFeedback, MoveAndContactResult, MoveAndContactAction
from std_msgs.msg import Bool

import tf

from baxter_interface import CHECK_VERSION

# Global variables
limb = ""
wait_time = 0

class COMMANDS:
    SLOW_MOVEMENT = 0
    MEDIUM_MOVEMENT = 1
    FAST_MOVEMENT = 2
    MEDIUM_MOVEMENT_ADAPTED = 3
    CALIBRATE_SENSOR = 4
    ACTIVATE_SENSOR = 5
    DEACTIVATE_SENSOR = 6

class Trajectory(object):
    def __init__(self, limb): # Initialize trajectory object
        ns = 'robot/limb/' + limb + '/' # Create SimpleActionClient
        self._client = actionlib.SimpleActionClient(ns + "follow_joint_trajectory",FollowJointTrajectoryAction)
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.05)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)
        gripper_topic = "/robot/end_effector/" + limb + "_gripper/gripper_action/goal"
        self.gripper_publisher = rospy.Publisher(gripper_topic, GripperCommandActionGoal, queue_size=10)

    def add_point(self, positions, velocities, accelerations, time): #Add point to trajectory target
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.velocities = copy(velocities)
        point.accelerations = copy(accelerations)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
        self._goal.trajectory.points = []
            
    def open_gripper(self): 
        gripperCommandMsg = GripperCommandActionGoal()
        gripperCommandMsg.goal.command.position = 100.0
        self.gripper_publisher.publish(gripperCommandMsg)

    def close_gripper(self): 
        gripperCommandMsg = GripperCommandActionGoal()
        gripperCommandMsg.goal.command.position = 0.0
        self.gripper_publisher.publish(gripperCommandMsg)

class MoveActionServer(object):
    # create messages that are used to publish feedback/result
    feedback = MoveAndContactFeedback()
    result = MoveAndContactResult()
    current_trajectory = None
    contact = False
    contact_subscriber = None
    
    def __init__(self, name):
        self.action_name = name # Create custom "move and detection" action server
        self.action_server = actionlib.SimpleActionServer(self.action_name, MoveAndContactAction, execute_cb=self.callback, auto_start = False)
        self.action_server.start()
        
    def callback(self, goal):
        goal = goal.traj # Receive array of trajectories
        print(goal.arm, "action server called")
        global wait_time
        rospy.sleep(wait_time)
        arm = goal.arm

        sensor_name = "/"+arm[0]+"_gripper_"+arm[0]+"_finger_"+goal.side+"_sensor"
        sensorActivatePub = rospy.Publisher(sensor_name + "_activate", Bool, queue_size=10) # Publisher to activate/deactivate contact detection
        sensorResetPub = rospy.Publisher(sensor_name + "_reset", Bool, queue_size=10) # Publisher reset sensor calibration, either if sensor is stuck on "contact" or because pose (and therefore gravity) changed
        contact_subscriber = rospy.Subscriber(sensor_name + "_response", Bool, self.cancel_current) # Cancel current trajectory if touch detected
        self.feedback.finished = False
        self.action_server.publish_feedback(self.feedback) # Set feedback to "not finished" (I don't use it)

        print(len(goal.control), "actions received")
        print("Actions received: ", goal.control)
        # CONFIGS
        time = [10, 5, 3.5] # Min time duration for slow and fast modes
        error = False
        self.contact = False
        j=0
        for i in range(len(goal.control)):
            print("Excecuting order",i+1,"/",len(goal.control))
            order = goal.control[i]
            
            self.current_trajectory = None
            if order == COMMANDS.ACTIVATE_SENSOR: # Activate sensor
                rospy.sleep(0.5)
                sensorActivatePub.publish(True)
                print("Sensor activated")
                continue
            elif order == COMMANDS.DEACTIVATE_SENSOR: # Deactivate sensor
                rospy.sleep(0.5)
                sensorActivatePub.publish(False)
                print("Sensor deactivated")
                continue
            elif order == COMMANDS.CALIBRATE_SENSOR: # Reset sensor
                rospy.sleep(0.5)
                sensorResetPub.publish(True)
                print("Sensor calibration reset")
                continue
            
            #If i reached here it means I have a trajectory to do...
            self.current_trajectory = Trajectory(arm) # Instance Trajectory class
            rospy.on_shutdown(self.current_trajectory.stop) # Stop immediately if ctrl+c
            limb_interface = baxter_interface.limb.Limb(arm) #what does this do?????
            
            times = [0.0] * len(goal.trajectory[j].joint_trajectory.points) # Times from beginning
            times = np.array(times)
            prev_joint = None

            for idx, point in enumerate(goal.trajectory[j].joint_trajectory.points): # Calculate proportional time for each point
                #input("This is point number " + str(idx))
                if prev_joint is None: # If this is first joint of chain time will be 1
                    #print("No previous joint, will check current")
                    prev_joint = [0.0] * len(point.positions)
                    current_joint = rospy.wait_for_message("robot/joint_states", JointState, timeout=3.0)
                    while len(current_joint.position) < 2:
                        current_joint = rospy.wait_for_message("robot/joint_states", JointState)
                    joint_orders = goal.trajectory[j].joint_trajectory.joint_names
                    for jdx, joint_name in enumerate(joint_orders):
                        joint_idx = current_joint.name.index(joint_name)
                        prev_joint[jdx] = current_joint.position[joint_idx]
                    self.current_trajectory.add_point(prev_joint, [], [], 0) # Add current point to traj
                    #print("First joint...", prev_joint)
                times[idx] = float(point.time_from_start.secs) + float(point.time_from_start.nsecs)/1000000000.0
            #print("Times: "+str(times))
            point_thresh = 20
            point_ratio = 1
            if order == COMMANDS.MEDIUM_MOVEMENT_ADAPTED: # Adjustable vel depending on points
                point_ratio = len(times) / point_thresh
                point_ratio = max(point_ratio,1) #Time_order max
                print("For this trajectory, time wil be",time[order]*point_ratio)
                order = COMMANDS.MEDIUM_MOVEMENT
            times *= point_ratio*time[order]/times[-1]
            times = list(times)
            #print("Times: "+str(times))

            for idx, point in enumerate(goal.trajectory[j].joint_trajectory.points):
                self.current_trajectory.add_point(point.positions, point.velocities, point.accelerations, times[idx])
                #print("Added point",point.positions)
            
            self.current_trajectory.start() #Start movement
            self.current_trajectory.wait(times[-1] + 5) #Wait for trajectory to finish or timeout of (5 sec + max time)
            traj_result = self.current_trajectory.result()
            if traj_result.error_code == 0 and not self.contact: # If everyhting finished ok
                print("Trajectory", j+1,"finished succesfully")
            #elif traj_result.error_code == -5:
                #print("Tolerance was violated but I can try to continue") # Ignre this common error...
            elif traj_result.error_code == -5 and j != len(goal.trajectory)-1: # If it breaks but it's not last trajectory...
                print("There was error but maybe still works...")
            else:
                if self.contact:
                    print("Trajectory ended because contact was detected")
                else:
                    print("Traj didnt finish well")
                    error = True # I show error
                break # Anyway i finish since all future traj are invalid
            self.current_trajectory = None # Preparing for new section of trajectory
            j += 1
        #rospy.sleep(1.5) # Wait 3 sec so hand stabilizes
        self.feedback.finished = True
        self.result.finished_succesfully = not error
        self.result.touch_detected = self.contact
        self.action_server.publish_feedback(self.feedback)
        self.action_server.set_succeeded(self.result)
        
        # When finished everyhting
        sensorActivatePub.publish(False)
        print("Orders completed")
        
    def cancel_current(self,msg):
        if self.current_trajectory is not None:
            print("TOUCH DETECTED, STOPPING")
            self.current_trajectory.stop()
            self.contact = True
        
def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-l',
        '--limb',
        required=True,
        choices=['left', 'right'],
        help='Send joint trajectory to which limb'
    )
    required.add_argument(
        '-w',
        '--wait_time',
        required=True,
        type=int,
        help='Wait time before moving real robot'
    )

    args = parser.parse_args(rospy.myargv()[1:])
    global limb
    global wait_time
    limb = args.limb
    wait_time = args.wait_time

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print("Enabling robot... ")
    rs.enable()
    
    server = MoveActionServer(limb + "_group/movement_manager")
    print(limb + " Action Server ready to receive orders")
    rospy.spin()

if __name__ == "__main__":
    main()
