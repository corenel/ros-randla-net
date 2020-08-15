cmake_minimum_required(VERSION 2.8.3)

project(ros_randla_net)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    )

generate_messages(DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
)

#catkin_install_python(PROGRAMS scripts/classify scripts/detect
#DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
