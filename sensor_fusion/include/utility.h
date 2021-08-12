#pragma once
#ifndef PARAMETER_H
#define PARAMETER_H

#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string.h>

#include "sensor_fusion/cloud_info.h"
#include "sensor_fusion/save_map.h"
#include "tic_toc.h"
#include "signal_handler.h"

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

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

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;
using namespace Eigen;

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

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))


using PointXYZIRT = VelodynePointXYZIRT;

typedef PointXYZIRPYT  PointTypePose;

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
		string odomTopic;

		// Frames
		string imuFrame;
		string mapFrame;
		string lidarFrame;
		string camFrame;
        string odometryFrame;

		// Feature
		float projectionErrorThres;
		int depthAssociateBinNumber;
		float pointFeatureDistThres;
		float stackPointTime;
		int lidarSkip;

		// Optimization
		float timeLag;

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
	    Eigen::Matrix3d rotL2I;
	    Eigen::Matrix3d rpyLidar2Imu;
	    Eigen::Vector3d transL2I;
	    Eigen::Vector3d imu2gpsTrans;
	    Eigen::Quaterniond extQRPY;

		// Transformation
		Eigen::Affine3f affineL2C;
		Eigen::Affine3f	affineL2I;
		Eigen::Affine3f	affineI2C;
		
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
	    float surroundingKeyframeSearchRadiusLarge;
		float surroundingKeyframeSearchRadiusSmall;
	    float surroundingKeyframeStackingThreshold;

	    
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
	    int WINDOW_SIZE;
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
		Eigen::Matrix3d rotI2C;
	    Eigen::Vector3d transI2C;

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
			nh.param<std::string>("fusion/odomTopic", odomTopic, "odom");

			nh.param<std::string>("fusion/lidarFrame", lidarFrame, "lidar_frame");
			nh.param<std::string>("fusion/imuFrame", imuFrame, "imu_frame");
			nh.param<std::string>("fusion/mapFrame", mapFrame, "map_frame");
            nh.param<std::string>("fusion/camFrame", camFrame, "cam_frame");
			nh.param<std::string>("fusion/odometryFrame", odometryFrame, "odom");

	        nh.param<int>("fusion/estimateExtrinsic", ESTIMATE_EXTRINSIC, 1);
			nh.param<float>("fusion/combined/projectionErrorThres", projectionErrorThres, 1.0);
			nh.param<int>("fusion/combined/depthAssociateBinNumber", depthAssociateBinNumber, 2);			
			nh.param<float>("fusion/combined/pointFeatureDistThres", pointFeatureDistThres, 1.0);
			nh.param<float>("fusion/combined/stackPointTime", stackPointTime, 1.0);
	        nh.param<int>("fusion/combined/lidarSkip", lidarSkip, 1);
			nh.param<vector<double>>("fusion/camera/rotImu2Cam", extImuCamRot, vector<double>());
	        nh.param<vector<double>>("fusion/camera/transImu2Cam", extImuCamTrans, vector<double>());
	        nh.param<float>("fusion/optimization/timeLag", timeLag, 2.0);

			rotI2C = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extImuCamRot.data(), 3, 3);
	        transI2C = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extImuCamTrans.data(), 3, 1);

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
			if (sensor_type == "velodyne")
			{
				sensor = SensorType::VELODYNE;
				printf("Velodyne imported\n");
			} 
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

	        nh.param<vector<double>>("fusion/laser/rotLidar2Imu", extRotV, vector<double>());
	        nh.param<vector<double>>("fusion/laser/extrinsicRPY", extRPYV, vector<double>());
	        nh.param<vector<double>>("fusion/laser/transLidar2Imu", extTransV, vector<double>());
	        rotL2I = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
	        // extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
	        transL2I = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
	        imu2gpsTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(imu2gpsTransV.data(), 3, 1);

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
	        nh.param<float>("fusion/laser/surroundingKeyframeSearchRadiusLarge", surroundingKeyframeSearchRadiusLarge, 50.0);
	        nh.param<float>("fusion/laser/surroundingKeyframeSearchRadiusSmall", surroundingKeyframeSearchRadiusSmall, 30.0);
	        nh.param<float>("fusion/laser/surroundingKeyframeStackingThreshold", surroundingKeyframeStackingThreshold, 50.0);

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
			nh.param<bool>("fusion/laser/savePCD", savePCD, false);
	        nh.param<string>("fusion/camera/calib", calibFile, ".yaml");
	        string configPath = ros::package::getPath("sensor_fusion") + "/config/";
	        string camPath = configPath + calibFile;
	        CAM_NAMES.push_back(camPath);

	        nh.param<float>("fusion/camera/focalLength", FOCAL_LENGTH, 460.0);
	        nh.param<int>("fusion/camera/windowSize", WINDOW_SIZE, 10);
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

			// Get affine transformation matrix
			Eigen::Vector3d ypr = rotL2I.eulerAngles(2, 1, 0);
			affineL2I = pcl::getTransformation(transL2I.x(), transL2I.y(), transL2I.z(), ypr.z(), ypr.y(), ypr.x());
			ypr = rotI2C.eulerAngles(2, 1, 0);
			affineI2C = pcl::getTransformation(transI2C.x(), transI2C.y(), transI2C.z(), ypr.z(), ypr.y(), ypr.x());
			affineL2C = affineL2I * affineI2C;

			// Test (Wonho)
			vector<double> rotVec{3.2830966986177872e-04, -9.9771742228419513e-01, 5.9009091480326889e-04, 
			-1.0695450518067135e-04, -5.9388257186973673e-04, -9.9817766066292501e-01, 
			9.9940368967948490e-01, 3.2140009005520810e-04, -1.2620811232126291e-02};

			vector<double> transVec{-0.13, -0.02, -0.18};

	        Eigen::Matrix3d rotC2L = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(rotVec.data(), 3, 3);
	        Eigen::Vector3d transC2L = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(transVec.data(), 3, 1);
			ypr = rotC2L.eulerAngles(2, 1, 0);
			Eigen::Affine3f affineC2L = pcl::getTransformation(transC2L.x(), transC2L.y(), transC2L.z(), ypr.z(), ypr.y(), ypr.x());
			affineL2C = affineC2L.inverse();
			
			// test
			printf("Utility.h\n");
			cout << affineL2C.matrix() << endl;
		}

        sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
        {
			Eigen::Matrix3d extRot = rotL2I;
            sensor_msgs::Imu imu_out = imu_in;
            // rotate acceleration
            Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
            acc = extRot * acc;
            imu_out.linear_acceleration.x = acc.x();
            imu_out.linear_acceleration.y = acc.y();
            imu_out.linear_acceleration.z = acc.z();
            // rotate gyroscope
            Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
            gyr = extRot * gyr;
            imu_out.angular_velocity.x = gyr.x();
            imu_out.angular_velocity.y = gyr.y();
            imu_out.angular_velocity.z = gyr.z();
            // rotate roll pitch yaw
			extQRPY = Eigen::Quaterniond(extRot.inverse());
            Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
            Eigen::Quaterniond q_final = q_from * extQRPY;
            imu_out.orientation.x = q_final.x();
            imu_out.orientation.y = q_final.y();
            imu_out.orientation.z = q_final.z();
            imu_out.orientation.w = q_final.w();

            if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
            {
                ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
                ros::shutdown();
            }

            return imu_out;
        }

		pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
		{
			pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

			int cloudSize = cloudIn->size();
			cloudOut->resize(cloudSize);

			Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
			
			#pragma omp parallel for num_threads(numberOfCores)
			for (int i = 0; i < cloudSize; ++i)
			{
				const auto &pointFrom = cloudIn->points[i];
				cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
				cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
				cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
				cloudOut->points[i].intensity = pointFrom.intensity;
			}
			return cloudOut;
		}
};

sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

PointTypePose trans2PointTypePose(Eigen::Affine3f transformIn)
{
	PointTypePose thisPose6D;
	pcl::getTranslationAndEulerAngles(transformIn, thisPose6D.x, thisPose6D.y, thisPose6D.z,
												   thisPose6D.roll, thisPose6D.pitch, thisPose6D.yaw);
	return thisPose6D;
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

double vectorDistance(Vector2d v1, Vector2d v2)
{
	return sqrt((v1.x()-v2.x())*(v1.x()-v2.x()) + (v1.y()-v2.y())*(v1.y()-v2.y()));
}

#endif