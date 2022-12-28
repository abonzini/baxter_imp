#!/bin/bash
rosrun baxter_examples send_urdf_fragment.py -f /home/simone/Desktop/catkin_ws/src/baxter_imp/xacros/left_end_effector.urdf.xacro -l left_hand -j left_gripper_base &
pidleft=$!
rosrun baxter_examples send_urdf_fragment.py -f /home/simone/Desktop/catkin_ws/src/baxter_imp/xacros/right_end_effector.urdf.xacro -l right_hand -j right_gripper_base &
pidright=$!
echo "Uploaded .xacro files, waiting..."
sleep 5
echo ".xacro setup complete"
