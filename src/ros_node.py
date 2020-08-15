import os
import rospy
import rospkg


class InferenceNode:

    def __init__(self):
        # params
        topic_pcl_sub = rospy.get_param('/ros_randla_net/topic/pcl_sub')
        topic_pcl_pub = rospy.get_param('/ros_randla_net/topic/pcl_pub')

        # model
        # subscription and publishing
        pass

    def callback(self, data):
        pass
