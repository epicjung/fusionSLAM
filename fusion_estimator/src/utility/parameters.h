#ifndef PARAMETER_H
#define PARAMETER_H

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
		string cloudTopic;
		string imuTopic;
		string imgTopic;

		// Frames
		string imuFrame;
		string mapFrame;
		string lidarFrame;
		string cameraFrame;

	    // GPS Settings
	    bool useImuHeadingInitialization;
	    bool useGpsElevation;
	    float gpsCovThreshold;
	    float poseCovThreshold;

	    // Save pcd
	    bool savePCD;
	    string savePCDDirectory;

	    // Lidar Sensor Configuration
	    SensorType sensor;
	    int N_SCAN;
	    int HORIZON_SCAN;
	    int downsampleRate;
	    float lidarMinRange;
	    float lidarMaxRange;

	    // IMU
	    float imuAccNoise;
	    float imuGyrNoise;
	    float imuAccBiasN;
	    float imuGyrBiasN;
	    float imuGravity;
	    float imuRPYWeight;
	    vector<double> extRotV;
	    vector<double> extRPYV;
	    vector<double> extTransV;
	    vector<double> imu2gpsTransV;
	    Eigen::Matrix3d extRot;
	    Eigen::Matrix3d extRPY;
	    Eigen::Vector3d extTrans;
	    Eigen::Vector3d imu2gpsTrans;
	    Eigen::Quaterniond extQRPY;

	    // LOAM
	    float edgeThreshold;
	    float surfThreshold;
	    int edgeFeatureMinValidNum;
	    int surfFeatureMinValidNum;

	    // voxel filter paprams
	    float odometrySurfLeafSize;
	    float mappingCornerLeafSize;
	    float mappingSurfLeafSize ;

	    float z_tollerance; 
	    float rotation_tollerance;

	    // CPU Params
	    int numberOfCores;
	    double mappingProcessInterval;

	    // Surrounding map
	    float surroundingkeyframeAddingDistThreshold; 
	    float surroundingkeyframeAddingAngleThreshold; 
	    float surroundingKeyframeDensity;
	    float surroundingKeyframeSearchRadius;
	    
	    // Loop closure
	    bool  loopClosureEnableFlag;
	    float loopClosureFrequency;
	    int   surroundingKeyframeSize;
	    float historyKeyframeSearchRadius;
	    float historyKeyframeSearchTimeDiff;
	    int   historyKeyframeSearchNum;
	    float historyKeyframeFitnessScore;

	    // global map visualization radius
	    float globalMapVisualizationSearchRadius;
	    float globalMapVisualizationPoseDensity;
	    float globalMapVisualizationLeafSize;

		ParamServer()
		{
			nh.param<std::string>("fusion/cloudTopic", cloudTopic, "points");
			nh.param<std::string>("fusion/imuTopic", imuTopic, "imu");
			nh.param<std::string>("fusion/imgTopic", imgTopic, "image");
			nh.param<std::string>("fusion/lidarFrame", lidarFrame, "lidar_frame");
			nh.param<std::string>("fusion/imuFrame", imuFrame, "imu_frame");
			nh.param<std::string>("fusion/mapFrame", mapFrame, "map_frame");

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
	        nh.param<int>("fusion/downsampleRate", downsampleRate, 1);
	        nh.param<float>("fusion/lidarMinRange", lidarMinRange, 1.0);
	        nh.param<float>("fusion/lidarMaxRange", lidarMaxRange, 1000.0);

	        nh.param<float>("fusion/imuAccNoise", imuAccNoise, 0.01);
	        nh.param<float>("fusion/imuGyrNoise", imuGyrNoise, 0.001);
	        nh.param<float>("fusion/imuAccBiasN", imuAccBiasN, 0.0002);
	        nh.param<float>("fusion/imuGyrBiasN", imuGyrBiasN, 0.00003);
	        nh.param<float>("fusion/imuGravity", imuGravity, 9.80511);
	        nh.param<float>("fusion/imuRPYWeight", imuRPYWeight, 0.01);
	        nh.param<vector<double>>("fusion/extrinsicRot", extRotV, vector<double>());
	        nh.param<vector<double>>("fusion/extrinsicRPY", extRPYV, vector<double>());
	        nh.param<vector<double>>("fusion/extrinsicTrans", extTransV, vector<double>());
	        nh.param<vector<double>>("fusion/imu2gpsTrans", imu2gpsTransV, vector<double>());
	        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
	        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
	        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
	        imu2gpsTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(imu2gpsTransV.data(), 3, 1);
	        extQRPY = Eigen::Quaterniond(extRPY);

	        nh.param<float>("fusion/edgeThreshold", edgeThreshold, 0.1);
	        nh.param<float>("fusion/surfThreshold", surfThreshold, 0.1);
	        nh.param<int>("fusion/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
	        nh.param<int>("fusion/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

	        nh.param<float>("fusion/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
	        nh.param<float>("fusion/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
	        nh.param<float>("fusion/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

	        nh.param<float>("fusion/z_tollerance", z_tollerance, FLT_MAX);
	        nh.param<float>("fusion/rotation_tollerance", rotation_tollerance, FLT_MAX);

	        nh.param<int>("fusion/numberOfCores", numberOfCores, 2);
	        nh.param<double>("fusion/mappingProcessInterval", mappingProcessInterval, 0.15);

	        nh.param<float>("fusion/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
	        nh.param<float>("fusion/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
	        nh.param<float>("fusion/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
	        nh.param<float>("fusion/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

	        nh.param<bool>("fusion/loopClosureEnableFlag", loopClosureEnableFlag, false);
	        nh.param<float>("fusion/loopClosureFrequency", loopClosureFrequency, 1.0);
	        nh.param<int>("fusion/surroundingKeyframeSize", surroundingKeyframeSize, 50);
	        nh.param<float>("fusion/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
	        nh.param<float>("fusion/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
	        nh.param<int>("fusion/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
	        nh.param<float>("fusion/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

	        nh.param<float>("fusion/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
	        nh.param<float>("fusion/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
	        nh.param<float>("fusion/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
		}

};

#endif