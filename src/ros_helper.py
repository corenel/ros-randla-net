import ctypes
import struct
from random import randint

import pcl
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


def random_color_gen():
    """
    Generates a random color

    :return: 3 elements, R, G, and B
    :rtype: list
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]


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
        PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1))

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


def XYZRGB_to_XYZ(XYZRGB_cloud):
    """
    Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)

    :param XYZRGB_cloud: A PCL XYZRGB point cloud
    :type XYZRGB_cloud: PointCloud_PointXYZRGB
    :return: A PCL XYZ point cloud
    :rtype: PointCloud_PointXYZ
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud


def XYZ_to_XYZRGB(XYZ_cloud, color, use_multiple_colors=False):
    """
    Converts a PCL XYZ point cloud to a PCL XYZRGB point cloud

    All returned points in the XYZRGB cloud will be the color indicated
    by the color parameter.

    :param XYZ_cloud: A PCL XYZ point cloud
    :type XYZ_cloud: PointCloud_XYZ
    :param color: 3-element list of integers [0-255,0-255,0-255]
    :type color: list
    :param use_multiple_colors: use more than one color
    :type use_multiple_colors: bool
    :return: A PCL XYZRGB point cloud
    :rtype: PointCloud_PointXYZRGB
    """
    XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
    points_list = []

    float_rgb = rgb_to_float(color) if not use_multiple_colors else None

    for idx, data in enumerate(XYZ_cloud):
        float_rgb = rgb_to_float(
            color[idx]) if use_multiple_colors else float_rgb
        points_list.append([data[0], data[1], data[2], float_rgb])

    XYZRGB_cloud.from_list(points_list)
    return XYZRGB_cloud


def XYZ_to_XYZI(XYZ_cloud, color, use_multiple_colors=False):
    XYZI_cloud = pcl.PointCloud_PointXYZI()
    points_list = []

    for idx, data in enumerate(XYZ_cloud):
        intensity = int(color[idx]) if use_multiple_colors else int(color)
        points_list.append([data[0], data[1], data[2], intensity])

    XYZI_cloud.from_list(points_list)
    return XYZI_cloud


def rgb_to_float(color):
    """
    Converts an RGB list to the packed float format used by PCL

    From the PCL docs:
    "Due to historical reasons (PCL was first developed as a ROS package),
     the RGB information is packed into an integer and casted to a float"

    :param color: 3-element list of integers [0-255,0-255,0-255]
    :type color: list
    :return: RGB value packed as a float
    :rtype: float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb


def float_to_rgb(float_rgb):
    """
    Converts a packed float RGB format to an RGB list

    :param float_rgb: RGB value packed as a float
    :type float_rgb: float
    :return: 3-element list of integers [0-255,0-255,0-255]
    :rtype: list
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = [r, g, b]

    return color
