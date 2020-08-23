import ctypes
import struct

import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


def ros_to_pcl(ros_cloud, field_type='xyzrgb'):
    """
    Converts a ROS PointCloud2 message to a pcl PointXYZRGB

    :param ros_cloud: ROS PointCloud2 message
    :type ros_cloud: PointCloud2
    :return: PCL XYZRGB point cloud
    :rtype: pcl.PointCloud_PointXYZRGB
    """
    points_list = []

    if field_type == 'xyzrgb':
        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        pcl_data = pcl.PointCloud_PointXYZRGB()
    elif field_type == 'xyz':
        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2]])
        pcl_data = pcl.PointCloud()
    else:
        raise NotImplementedError(
            'Unsupported field type: {}'.format(field_type))

    pcl_data.from_list(points_list)

    return pcl_data


def pcl_to_ros(pcl_array, frmae_id='world', timestamp=None):
    """
    Converts a ROS PointCloud2 message to a pcl PointXYZRGB

    :param timestamp: message timestamp
    :type timestamp: rospy.Time
    :param frmae_id: frame id
    :type frmae_id: str
    :param pcl_array: A PCL XYZRGB point cloud
    :type pcl_array: PointCloud_PointXYZRGB
    :return: A ROS point cloud
    :rtype: PointCloud2
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now() if timestamp is None else timestamp
    ros_msg.header.frame_id = frmae_id

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(
        PointField(name="rgb", offset=16, datatype=PointField.FLOAT32,
                   count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(
            struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r,
                        0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg
