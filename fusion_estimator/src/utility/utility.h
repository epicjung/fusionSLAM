
#ifndef UTILITY_H
#define UTILITY_H

#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <deque>
#include <opencv/cv.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

typedef pcl::PointXYZI PointType;

class Utility
{
	public:

		template<typename T>
		static void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
		{
		    double imuRoll, imuPitch, imuYaw;
		    tf::Quaternion orientation;
		    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
		    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

		    *rosRoll = imuRoll;
		    *rosPitch = imuPitch;
		    *rosYaw = imuYaw;
		}

		template<typename T>
		static void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
		{
			*angular_x = thisImuMsg->angular_velocity.x;
			*angular_y = thisImuMsg->angular_velocity.y;
    		*angular_z = thisImuMsg->angular_velocity.z;
		}

		static float pointDistance(PointType p)
		{
		    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
		}

		static sensor_msgs::PointCloud2 toROSPointCloud(pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
		{
		    sensor_msgs::PointCloud2 tempCloud;
		    pcl::toROSMsg(*thisCloud, tempCloud);
		    tempCloud.header.stamp = thisStamp;
		    tempCloud.header.frame_id = thisFrame;
		    return tempCloud;
		}
};

#endif

