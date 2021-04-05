#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

// #include <opencv2/core/eigen.hpp>
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


struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

using PointXYZIRT = VelodynePointXYZIRT;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER };

using namespace std;

class ParamServer
{
	public:
		ros::NodeHandle nh;

		// Topics
		string cloud_topic;
		string imu_topic;
		string img_topic;

		// Frames
		string imu_frame;
		string map_frmae;
		string lidar_frame;
		string camera_frame;

		// LiDAR
		SensorType sensor;
		int N_SCAN;
		int HORIZON_SCAN;
		int downsample_rate;
		float lidar_min_range;
		float lidar_max_range;

		ParamServer()
		{
			nh.param<std::string>("fusion/pointCloudTopic", cloud_topic, "points");
			nh.param<std::string>("fusion/imuTopic", imu_topic, "imu");
			nh.param<std::string>("fusion/imgTopic", img_topic, "image");

			string sensor_type;
			nh.param<std::string>("fusion/sensor", sensor_type, "ouster");
			if (sensor_type == "velodyne") sensor = SensorType::VELODYNE;
			else if (sensor_type == "ouster") sensor = SensorType::OUSTER;
			else 
			{
				ROS_ERROR_STREAM("Invalid sensor type.");
				ros::shutdown();
			}

			nh.param<int>("fusion/N_SCAN", N_SCAN, 16);
			nh.param<int>("fusion/HORIZON_SCAN", HORIZON_SCAN, 1800);
	        nh.param<int>("fusion/downsampleRate", downsample_rate, 1);
	        nh.param<float>("fusion/lidarMinRange", lidar_min_range, 1.0);
	        nh.param<float>("fusion/lidarMaxRange", lidar_max_range, 1000.0);
		}
};