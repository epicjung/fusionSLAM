#ifndef PARAMETER_H
#define PARAMETER_H

#include <ros/ros.h>
#include <ros/package.h>
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

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

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

		//#define UNIT_SPHERE_ERROR

	    string calibFile;
		float FOCAL_LENGTH;
	    // int WINDOW_SIZE;
	    int NUM_OF_F;

		double INIT_DEPTH;
		double MIN_PARALLAX;
		double KEYFRAME_PARALLAX;
		int ESTIMATE_EXTRINSIC;

		double ACC_N, ACC_W;
		double GYR_N, GYR_W;

		Eigen::Vector3d G;
		double gNorm;

		vector<string> CAM_NAMES;

		double BIAS_ACC_THRESHOLD;
		double BIAS_GYR_THRESHOLD;
		double SOLVER_TIME;
		int NUM_ITERATIONS;
		std::string EX_CALIB_RESULT_PATH;
		std::string VINS_RESULT_PATH;
		std::string OUTPUT_FOLDER;
		std::string IMU_TOPIC;
		double TD;
		int ESTIMATE_TD;

		vector<double> extImuCamRot;
		vector<double> extImuCamTrans;
		vector<Eigen::Matrix3d> RIC;
		vector<Eigen::Vector3d> TIC;

		int ROW, COL;
		int NUM_OF_CAM;
		int STEREO;
		int USE_IMU;
		int MULTIPLE_THREAD;
		// pts_gt for debug purpose;
		map<int, Eigen::Vector3d> pts_gt;

		int MAX_CNT;
		int MIN_DIST;
		double F_THRESHOLD;
		int SHOW_TRACK;
		int FLOW_BACK;


		ParamServer()
		{
			nh.param<std::string>("fusion/cloudTopic", cloudTopic, "points");
			nh.param<std::string>("fusion/imuTopic", imuTopic, "imu");
			nh.param<std::string>("fusion/imgTopic", imgTopic, "image");
			nh.param<std::string>("fusion/lidarFrame", lidarFrame, "lidar_frame");
			nh.param<std::string>("fusion/imuFrame", imuFrame, "imu_frame");
			nh.param<std::string>("fusion/mapFrame", mapFrame, "map_frame");

	        nh.param<int>("fusion/estimateExtrinsic", ESTIMATE_EXTRINSIC, 1);
	        nh.param<vector<double>>("fusion/imuCamRotation/data", extImuCamRot, vector<double>());
	        nh.param<vector<double>>("fusion/imuCamTranslation/data", extImuCamTrans, vector<double>());
	        
		    if (ESTIMATE_EXTRINSIC == 2)
		    {
		        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
		        RIC.push_back(Eigen::Matrix3d::Identity());
		        TIC.push_back(Eigen::Vector3d::Zero());
		        // EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
		    }
		    else 
		    {
		        if ( ESTIMATE_EXTRINSIC == 1)
		        {
		            ROS_WARN(" Optimize extrinsic param around initial guess!");
		            // EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
		        }
		        if (ESTIMATE_EXTRINSIC == 0)
		            ROS_WARN(" fix extrinsic param ");

				RIC.push_back(Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extImuCamRot.data(), 3, 3));
				TIC.push_back(Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extImuCamTrans.data(), 3, 1));
		    } 

			string sensor_type;
			nh.param<std::string>("fusion/laser/sensor", sensor_type, "ouster");
			if (sensor_type == "velodyne") sensor = SensorType::VELODYNE;
			else if (sensor_type == "ouster") sensor = SensorType::OUSTER;
			else 
			{
				ROS_ERROR_STREAM("Invalid sensor type.");
				ros::shutdown();
			}

			nh.param<int>("fusion/laser/N_SCAN", N_SCAN, 16);
			nh.param<int>("fusion/laser/HORIZON_SCAN", HORIZON_SCAN, 1800);
	        nh.param<int>("fusion/laser/downsampleRate", downsampleRate, 1);
	        nh.param<float>("fusion/laser/lidarMinRange", lidarMinRange, 1.0);
	        nh.param<float>("fusion/laser/lidarMaxRange", lidarMaxRange, 1000.0);

	        nh.param<float>("fusion/imu/imuAccNoise", imuAccNoise, 0.01);
	        nh.param<float>("fusion/imu/imuGyrNoise", imuGyrNoise, 0.001);
	        nh.param<float>("fusion/imu/imuAccBiasN", imuAccBiasN, 0.0002);
	        nh.param<float>("fusion/imu/imuGyrBiasN", imuGyrBiasN, 0.00003);
	        nh.param<float>("fusion/imu/imuGravity", imuGravity, 9.80511);
	        nh.param<float>("fusion/imu/imuRPYWeight", imuRPYWeight, 0.01);
	        nh.param<double>("fusion/imu/gNorm", gNorm, 9.8);
	        G.z() = gNorm;

	        nh.param<vector<double>>("fusion/imu/imu2gpsTrans", imu2gpsTransV, vector<double>());


	        nh.param<vector<double>>("fusion/laser/extrinsicRot", extRotV, vector<double>());
	        nh.param<vector<double>>("fusion/laser/extrinsicRPY", extRPYV, vector<double>());
	        nh.param<vector<double>>("fusion/laser/extrinsicTrans", extTransV, vector<double>());
	        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
	        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
	        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
	        imu2gpsTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(imu2gpsTransV.data(), 3, 1);
	        extQRPY = Eigen::Quaterniond(extRPY);

	        nh.param<float>("fusion/laser/edgeThreshold", edgeThreshold, 0.1);
	        nh.param<float>("fusion/laser/surfThreshold", surfThreshold, 0.1);
	        nh.param<int>("fusion/laser/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
	        nh.param<int>("fusion/laser/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

	        nh.param<float>("fusion/laser/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
	        nh.param<float>("fusion/laser/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
	        nh.param<float>("fusion/laser/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

	        nh.param<float>("fusion/laser/z_tollerance", z_tollerance, FLT_MAX);
	        nh.param<float>("fusion/laser/rotation_tollerance", rotation_tollerance, FLT_MAX);

	        nh.param<int>("fusion/laser/numberOfCores", numberOfCores, 2);
	        nh.param<double>("fusion/laser/mappingProcessInterval", mappingProcessInterval, 0.15);

	        nh.param<float>("fusion/laser/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
	        nh.param<float>("fusion/laser/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
	        nh.param<float>("fusion/laser/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
	        nh.param<float>("fusion/laser/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

	        nh.param<bool>("fusion/laser/loopClosureEnableFlag", loopClosureEnableFlag, false);
	        nh.param<float>("fusion/laser/loopClosureFrequency", loopClosureFrequency, 1.0);
	        nh.param<int>("fusion/laser/surroundingKeyframeSize", surroundingKeyframeSize, 50);
	        nh.param<float>("fusion/laser/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
	        nh.param<float>("fusion/laser/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
	        nh.param<int>("fusion/laser/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
	        nh.param<float>("fusion/laser/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

	        nh.param<float>("fusion/laser/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
	        nh.param<float>("fusion/laser/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
	        nh.param<float>("fusion/laser/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

	        nh.param<string>("fusion/camera/calib", calibFile, ".yaml");
	        string configPath = ros::package::getPath("fusion_estimator") + "/config/";
	        string camPath = configPath + calibFile;
	        CAM_NAMES.push_back(camPath);

	        nh.param<float>("fusion/camera/focalLength", FOCAL_LENGTH, 460.0);
	        // nh.param<int>("fusion/camera/windowSize", WINDOW_SIZE, 10);
	        nh.param<int>("fusion/camera/numOfF", NUM_OF_F, 1000);

	        nh.param<double>("fusion/camera/initDepth", INIT_DEPTH, 1.0);
	        nh.param<double>("fusion/camera/keyframeParallax", KEYFRAME_PARALLAX, 1.0);
	        
	        MIN_PARALLAX = KEYFRAME_PARALLAX / FOCAL_LENGTH;

	        nh.param<double>("fusion/camera/solverTime", SOLVER_TIME, 0.0);
	        nh.param<int>("fusion/camera/numIterations", NUM_ITERATIONS, 2);
	        nh.param<double>("fusion/camera/TD", TD, 1.0);
	        nh.param<int>("fusion/camera/estimateTD", ESTIMATE_TD, 1);
	        nh.param<int>("fusion/camera/imageHeight", ROW, 1);
	        nh.param<int>("fusion/camera/imageWidth", COL, 1);
	        
	        nh.param<int>("fusion/camera/maxCnt", MAX_CNT, 100);
	        nh.param<int>("fusion/camera/minDist", MIN_DIST, 100);
	        nh.param<double>("fusion/camera/FTheshold", F_THRESHOLD, 10.0);
	        nh.param<int>("fusion/camera/showTrack", SHOW_TRACK, 1);
	        nh.param<int>("fusion/camera/flowBack", FLOW_BACK, 1);

	        printf("Parameters have been loaded\n");
		}

};

#endif