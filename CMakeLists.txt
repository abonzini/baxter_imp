cmake_minimum_required(VERSION 3.0.2)
project(baxter_imp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
  moveit_msgs
  sensor_msgs
  controller_manager
  roslaunch
  trajectory_msgs
  arduino_magnetic_sensor
)

roslaunch_add_file_check(launch)

add_message_files(DIRECTORY msg)

add_service_files(DIRECTORY srv)

add_action_files(DIRECTORY action)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  moveit_msgs
  sensor_msgs
)

catkin_package(
  # INCLUDE_DIRS include
  LIBRARIES baxter_imp
  CATKIN_DEPENDS actionlib_msgs rospy std_msgs
  DEPENDS system_lib
  DEPENDS message_runtime
)

catkin_install_python(PROGRAMS
  scripts/tactile_sensor.py
  scripts/motion_planner.py
  scripts/joint_trajectory_client.py
  scripts/ExploreNoSym.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
