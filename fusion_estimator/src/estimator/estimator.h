#pragma once 

#include <thread>
#include <mutex>
#include <queue>
#include "fusion_estimator/CloudInfo.h"

#include "../utility/utility.h"
#include "../utility/parameters.h"
#include "../featureExtractor/feature_extractor.h"
#include "../featureTracker/feature_tracker.h"

using namespace std;
using namespace Eigen;


class Estimator : public ParamServer
{
	public:
		Estimator();
		~Estimator();
		void setParameter();
		void allocateMemory();
		void resetImageParameters();
		void resetLaserParameters();
		// data processing
		void inputCloud(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
		void inputOdom(const nav_msgs::Odometry::ConstPtr &odom_msg);
		void inputIMU(const sensor_msgs::ImuConstPtr &imu_msg);
		void inputImage(double t, const cv::Mat &img);

		bool IMUAvailable(const double t);
		bool getIMUInterval(double start, double end, vector<sensor_msgs::Imu> &imu_vec);

		bool cachePointCloud();
		void imuDeskew(vector<sensor_msgs::Imu> &imu_vec);
		void odomDeskew();
		void projectPointCloud();
		void cloudExtraction();
		void edgeSurfExtraction();
		PointType deskewPoint(PointType *point, double rel_time);
    	void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);
	    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);

		// void processIMU(double t, double dt, const Vector3d &acc, const Vector3d &gyr);
		void processMeasurements();

		// ros::Publisher pub_deskew;
		// ros::Publisher pub_cloudInfo;

		queue<sensor_msgs::Imu> imuBuf;
		queue<sensor_msgs::PointCloud2> cloudBuf;
		queue<fusion_estimator::CloudInfo> cloudInfoBuf;
    	queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
		deque<nav_msgs::Odometry> odomBuf;

		mutex mBuf;
		mutex mProcess;
		mutex mPropagate;
		thread processThread;

		FeatureExtractor featureExtractor;
		FeatureTracker featureTracker;

		bool initThreadFlag;
		bool validPointFlag;
		bool firstPointFlag;
		bool odomDeskewFlag;
		int deskewFlag;
    	cv::Mat rangeMat;

		sensor_msgs::PointCloud2 currentCloud;
		pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
		pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
		pcl::PointCloud<PointType>::Ptr fullCloud;
		pcl::PointCloud<PointType>::Ptr extractedCloud;
		fusion_estimator::CloudInfo cloudInfo;
		std_msgs::Header cloudHeader;
		double scanStartTime;
		double scanEndTime;

		int inputImageCnt;

		double lastImuTime;
		int imuPointerCur;
		Vector3d imuR[1000];
		Vector3d imuP[1000]; 

		double imuTime[1000];		
	    float odomIncreX;
	    float odomIncreY;
	    float odomIncreZ;
        Eigen::Affine3f transStartInverse;
};