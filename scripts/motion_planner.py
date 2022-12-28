import rospy
import time
import argparse
import sys
import copy
import moveit_commander
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
from geometry_msgs.msg import Quaternion, Pose, PoseStamped
from baxter_imp.msg import *
from baxter_imp.srv import *
import tf2_ros
import tf
from tf import transformations as ts
import numpy as np

lineLengthDivision = 0.015 # Segment length limit when planning cartesian path
plannerInterp = 0.01
goalTolerance = 0.01 # Tolerance

class MotionPlanner:
    def __init__(self, limb):
        self.limb = limb
        self.baxterJointNames = [self.limb + '_' + joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
        
        print("Initializing motion planner...")
        joint_state_topic = ['joint_states:=/robot/joint_states']
        moveit_commander.roscpp_initialize(joint_state_topic)
        
        print("Initializing commander classes")
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.limb + "_arm")
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(2)
        
        ps = PoseStamped()
        ps.pose = self.VectorToPose([0.665, -0.0,  -0.2],[0,0,0,1])
        ps.header.frame_id = "world"
        self.cube_pose = ps
        self.cube_size = (0.6,0.6,0.4)
        self.scene.add_box("tripod", self.cube_pose, size = self.cube_size)
        self.cube_size = (0,0,0)

        print("Initializing tf2 buffer")
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.sleep(2)
        
        self.update_tip_transform() # Gets relative tf between end effector and finger_tip
        self.update_inner_transform() # Gets relative tf between end effector and finger_inner
        
        print("Initialization complete.")
        #Print all for debug purposes
        #eef_link = self.group.get_end_effector_link()
        #print("End effector link: ", eef_link)
        #planning_frame = self.group.get_planning_frame()
        #print("Planning frame:", planning_frame)
        #group_names = self.robot.get_group_names()
        #print("Planning Groups:", group_names)
        #state = self.robot.get_current_state()
        #print("State:", state)
        #pose = self.group.get_current_pose()
        #print("Pose:", pose)
        #joints = self.group.get_current_joint_values()
        #print("Joints:", joints)
        self.group.clear_pose_targets()
    def SetBoundsMsg(self,msg):
        ps = PoseStamped()
        ps.pose = self.VectorToPose(msg.sphere_center,[0,0,0,1])
        ps.header.frame_id = "world"
        self.cube_pose = ps
        self.cube_size = (2*msg.radius, 2*msg.radius, 2*msg.height)
        print("Bounds for exploration area have been updated in "+self.limb +"group planner:",msg.sphere_center)
    def update_tip_transform(self):
        # Tf (usually static) frame between real end effector and finger tip (for calculations)
        tStamp = self.tfBuffer.lookup_transform(self.limb+'_gripper', self.limb[0]+"_gripper_"+self.limb[0]+"_finger_tip", rospy.Time(0),rospy.Duration(5.0)) 
        (self.tip_trans, self.tip_rot) = self.TransformToVector(tStamp.transform)
    def update_inner_transform(self):
        # Tf (usually static) frame between real end effector and finger inner (for calculations)
        tStamp = self.tfBuffer.lookup_transform(self.limb+'_gripper', self.limb[0]+"_gripper_"+self.limb[0]+"_finger_inner",rospy.Time(0),rospy.Duration(5.0)) 
        (self.inner_trans, self.inner_rot) = self.TransformToVector(tStamp.transform)
    def ReverseTransform(self, pos, orient, transform_pos, transform_orient):
        # if link A -> (tf) -> B, this function moves a point (pos,orient) from B -> (tf**-1) -> A
        # Used to plan a desired fingertip to end effector pose
        PoseMatrix = ts.concatenate_matrices(ts.translation_matrix(pos), ts.quaternion_matrix(orient))
        TransformMatrix = ts.concatenate_matrices(ts.translation_matrix(transform_pos), ts.quaternion_matrix(transform_orient))
        InvertedTtransform = ts.inverse_matrix(TransformMatrix)
    
        ResultingPose = np.dot(PoseMatrix, InvertedTtransform)
        res_pos = ts.translation_from_matrix(ResultingPose)
        res_orient = ts.quaternion_from_matrix(ResultingPose)
        return res_pos, res_orient
    # Conversion from tf.ts, and Pose object to np vector and vice versa
    def TransformToVector(self,transform):
        return np.array([transform.translation.x, transform.translation.y, transform.translation.z]), np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    def PoseToVector(self,pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z]), np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    def VectorToPose(self, point, orientation):
        res = Pose()
        res.position.x = point[0]
        res.position.y = point[1]
        res.position.z = point[2]
        res.orientation.x = orientation[0]
        res.orientation.y = orientation[1]
        res.orientation.z = orientation[2]
        res.orientation.w = orientation[3]
        return res
    # Planning stright lines
    def PlanCartesian(self, destination_position, destination_orientation, initial_position , initial_orientation, initial_joints):
        print("Will plan cartesian (straight line) trajectory on "+self.limb+" arm")
        waypoints = []
        # If no initial config, current end effector location is used
        initial_pose = self.VectorToPose(initial_position, initial_orientation)

        # Split path into parts of lineLengthDivision separation. This is to ensure a cartesian traj of a straight line
        direction = destination_position-initial_position
        distance = np.linalg.norm(direction)
        n = 1
        if distance > lineLengthDivision:
            n = int(round(distance/lineLengthDivision))
            direction /= n
        for i in range(1,n):
            dx = direction * i
            next_pose = copy.deepcopy(initial_pose)
            next_pose.position.x += dx[0]
            next_pose.position.y += dx[1]
            next_pose.position.z += dx[2]
            waypoints.append(next_pose)
        destination_pose = self.VectorToPose(destination_position, destination_orientation)
        waypoints.append(destination_pose) # Final waypoint is the goal
        # Result is a waypoints array with separation of lineLengthDivision and final point. This only works properly if the initial pose is equal to destination pose
        
        start_joint_state = JointState()
        start_joint_state.name = self.baxterJointNames
        start_joint_state.position = initial_joints # Create my initial robot state, with initial joints
        start_robot_state = RobotState()
        start_robot_state.joint_state = start_joint_state
        self.group.set_start_state(start_robot_state)
        self.group.set_goal_tolerance(goalTolerance)
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, plannerInterp, 0.0)
        self.group.clear_pose_targets()
        success = (fraction>0.80) # Consider path a success if 80% i guess would make it good enough?
        return plan, success
    def PlanCartesianUnwrapper(self,req):
        response = PlannerServiceResponse()
        if not req.initial_position or not req.initial_orientation or not req.initial_joints:
            req.initial_joints = self.group.get_current_joint_values()
            initial_pose = self.group.get_current_pose().pose
            init_endeff_position, init_endeff_orientation = self.PoseToVector(initial_pose)
        else:
            init_endeff_position, init_endeff_orientation = self.GetEndeffTarget(req.initial_position, req.initial_orientation, req.sensor_side)

        #Get real pose of end effector
        dest_endeff_position, dest_endeff_orientation = self.GetEndeffTarget(req.destination_position, req.destination_orientation, req.sensor_side)
        #Then plan
        traj, succ = self.PlanCartesian(dest_endeff_position, dest_endeff_orientation, init_endeff_position, init_endeff_orientation, req.initial_joints)
        response.success = succ
        response.planned_trajectory = traj
        return response
    # Plan to a specific pose
    def PlanToPose(self, destination_position, destination_orientation, initial_joints = None):
        if not initial_joints:
            initial_joints = self.group.get_current_joint_values()
        
        self.scene.add_box("exploration_bounds", self.cube_pose, size = self.cube_size) # Add this so it plans outside the sphere without touching anything
        rospy.sleep(0.33)
        #print("added obstacle sphere")
        
        start_joint_state = JointState()
        start_joint_state.name = self.baxterJointNames
        start_joint_state.position = initial_joints # Create my initial robot state, with initial joints
        start_robot_state = RobotState()
        start_robot_state.joint_state = start_joint_state
        self.group.set_start_state(start_robot_state)
        self.group.set_goal_tolerance(goalTolerance)
        #print("Set starting joint state")
        
        destination_pose = self.VectorToPose(destination_position, destination_orientation)
        #input("Destination pose:"+str(destination_pose))
        
        self.group.set_pose_target(destination_pose)
        plan = self.group.plan()
        self.group.clear_pose_targets()
        
        success = plan[0]
        plan = plan[1]
        
        self.scene.remove_world_object("exploration_bounds")
        return plan, success
    def PlanToPoseUnwrapper(self,req):
        response = PlannerServiceResponse()
        #print("Received a request to plan. position:",req.destination_position,"orientation:",req.destination_orientation,"side:",req.sensor_side,"initial joints:",req.initial_joints)
        #Get real pose of end effector
        endeff_position, endeff_orientation = self.GetEndeffTarget(req.destination_position, req.destination_orientation, req.sensor_side)
        #print("Want to go to pos:",req.destination_position,"orient", req.destination_orientation, "with", req.sensor_side)
        #print("For this, the gripper wil go to",endeff_position,"with orient",endeff_orientation)
        #input()
        #Then plan
        traj, succ = self.PlanToPose(endeff_position, endeff_orientation, initial_joints = req.initial_joints)
        response.success = succ
        response.planned_trajectory = traj
        return response
    def PlanToNeutralJoint(self, req):
        self.scene.add_box("exploration_bounds", self.cube_pose, size = self.cube_size) # Add this so it plans outside the sphere without touching anything
        rospy.sleep(0.33)
        
        self.group.set_start_state_to_current_state()
        if self.limb == 'right':
            hardcodedJointPosition = [-0.12463593901568987, -1.0105098440195164, 0.6665146523362123, 1.562359432461294, -0.3351748021529629, 1.1520195717019457, 0.0]
            #hardcodedJointPosition = [0.08283496254581234, -1.002456444883118, 1.188068120217253, 1.9397187062811057, -0.6691991187150117, 1.0323690702468835, 0.4973932704718454]
        else:
            hardcodedJointPosition = [0.12463593901568987, -1.0105098440195164, -0.6665146523362123, 1.562359432461294, 0.3351748021529629, 1.1520195717019457, 0.0]
            #hardcodedJointPosition = [-0.08283496254581234, -1.002456444883118, -1.188068120217253, 1.9397187062811057, 0.6691991187150117, 1.0323690702468835, -0.4973932704718454]
        
        target_joint_state = JointState()
        target_joint_state.name = self.baxterJointNames
        target_joint_state.position = hardcodedJointPosition
        
        self.group.set_joint_value_target(target_joint_state)
        plan = self.group.plan()
        
        response = PlannerServiceResponse()
        
        self.group.clear_pose_targets()
        response.success = plan[0]
        response.planned_trajectory = plan[1]
        self.scene.remove_world_object("exploration_bounds")
        return response
    def GetEndeffTarget(self, dest_pos, dest_orient, sensor_side):
        if sensor_side == "tip":
            return self.ReverseTransform(dest_pos, dest_orient, self.tip_trans, self.tip_rot)
        elif sensor_side == "inner":
            return self.ReverseTransform(dest_pos, dest_orient, self.inner_trans, self.inner_rot)
        else:
            return dest_pos, dest_orient
     
def main():
  # Initialize node
  rospy.init_node('motion_planner_service_node')

  # Parse argument from launch file
  arg_fmt = argparse.RawDescriptionHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                    description=main.__doc__)
  required = parser.add_argument_group('required arguments')
  required.add_argument(
      '-l', '--limb', required=True, type=str,
      help='limb parameter [left, right]'
  )

  args = parser.parse_args(rospy.myargv()[1:])
  limb = args.limb

  # Define motion planner object with argument parsed
  motion_planner = MotionPlanner(limb)
  rospy.Service('line_trajectory_planner', PlannerService, motion_planner.PlanCartesianUnwrapper) # Set service for linear movement
  rospy.Service('pose_planner', PlannerService, motion_planner.PlanToPoseUnwrapper) # Set service for pose movemenr
  rospy.Service('neutral_planner', PlannerService, motion_planner.PlanToNeutralJoint) # Self service to plan to joint position
  rospy.Subscriber("exploration_bounds", SetBoundsMsg, motion_planner.SetBoundsMsg)
  
  print("Ready to plan")
  rospy.spin()


if __name__ == "__main__":
    main()
    


