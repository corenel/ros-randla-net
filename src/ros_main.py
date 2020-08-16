#!/usr/bin/env python

from __future__ import print_function

import roslib

roslib.load_manifest('ros_randla_net')

import sys
import rospy
from ros_node import InferenceNode


def main(args):
    rospy.init_node('~', anonymous=True)
    node = InferenceNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
