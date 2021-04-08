#pragma once 

#include <thread>
#include <mutex>
#include <queue>
#include "fusion_estimator/CloudInfo.h"

#include "../utility/utility.h"
#include "../utility/parameters.h"
#include "../featureExtractor/feature_extractor.h"
#include "../featureTracker/feature_tracker.h"
// #include "../mapOptimizer/map_optimizer.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using namespace std;
using namespace Eigen;
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
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

typedef PointXYZIRPYT  PointTypePose;

const int WINDOW_SIZE = 10;
const int queueLength = 2000; 

class Estimator : public ParamServer
{
	public:
		Estimator();
		~Estimator();
		void setParameter();
		void allocateMemory();
		void resetImageParameters();
		void resetLaserParameters();
		void resetOptimization();
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
		void updateInitialGuess();
		void extractSurroundingKeyFrames();
		void downsampleCurrentScan();
		void scan2MapOptimization();
		bool isCloudKeyframe();
		void addOdomFactor();
		void optimize();
		void saveKeyframe(int type);
		void publishOdometry();

		// miscell
		Affine3f trans2Affine3f(Vector3d XYZ, Vector3d RPY);
		Pose3 trans2gtsamPose(Vector3d XYZ, Vector3d RPY);


		// ros::Publisher pub_deskew;
		// ros::Publisher pub_cloudInfo;

		queue<sensor_msgs::Imu> imuBuf;
		queue<sensor_msgs::PointCloud2> cloudBuf;
		queue<fusion_estimator::CloudInfo> cloudInfoBuf;
    	queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
		deque<nav_msgs::Odometry> odomBuf;
		fusion_estimator::CloudInfo cloudInfoIn; 

		mutex mBuf;
		mutex mProcess;
		mutex mPropagate;
		thread processThread;

		FeatureExtractor featureExtractor;
		FeatureTracker featureTracker;
		// MapOptimizer mapOptimizer;

		bool initThreadFlag;
		bool initSystemFlag;
		bool firstPointFlag;
		bool initImuFlag;
		bool odomDeskewFlag;
		int deskewFlag;
    	cv::Mat rangeMat;

	    // gtsam
	    gtsam::NonlinearFactorGraph factorGraph;
	    gtsam::ISAM2 *optimizer;
	    gtsam::Values initialEstimate;
	    gtsam::Values isamCurrentEstimate;
    	Eigen::MatrixXd poseCovariance;

    	// pointcloud
		sensor_msgs::PointCloud2 currentCloud;
		pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
		pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
		pcl::PointCloud<PointType>::Ptr fullCloud;
		pcl::PointCloud<PointType>::Ptr extractedCloud;
		fusion_estimator::CloudInfo cloudInfoOut;
		std_msgs::Header cloudHeader;
		double scanStartTime;
		double scanEndTime;
		int cloudFrameCount;

		// image 
		int inputImageCnt;
		int imgFrameCount;
		double prevImgTime;
		double td;
		Vector3d g;
		Matrix3d ric[2];
		Vector3d tic[2];

		// imu
		double lastImuTime;
		int imuPointerCur;
		Vector3d imuR[queueLength];
		Vector3d imuP[queueLength]; 
		double imuTime[queueLength];
		vector<sensor_msgs::Imu> initImuVector;

		// odom
	    float odomIncreX;
	    float odomIncreY;
	    float odomIncreZ;
        Eigen::Affine3f transStartInverse;
    	Eigen::Affine3f increOdomFront;
    	Eigen::Affine3f increOdomBack;

        // map optimization
        ros::Time rosTimeCloudInfo;
        double timeCloudInfoIn;
        pcl::PointCloud<PointType>::Ptr laserCloudCorner; // corner feature set from odoOptimization
    	pcl::PointCloud<PointType>::Ptr laserCloudSurf; // surf feature set from odoOptimization
    	pcl::PointCloud<PointType>::Ptr laserCloudCornerDS;
    	pcl::PointCloud<PointType>::Ptr laserCloudSurfDS;
    	vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    	vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    	pcl::VoxelGrid<PointType> downSizeFilterCorner;
    	pcl::VoxelGrid<PointType> downSizeFilterSurf;
    	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
	    int laserCloudCornerFromMapDSNum;
	    int laserCloudSurfFromMapDSNum;
	    int laserCloudCornerDSNum;
	    int laserCloudSurfDSNum;


    	// State
    	Vector3d latestXYZ, latestRPY, latestV, latestBa, latestBg; 
    	
    	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    	Vector3d camKeyPos[(WINDOW_SIZE + 1)];
    	Vector3d camKeyVel[(WINDOW_SIZE + 1)];
    	Matrix3d camKeyRot[(WINDOW_SIZE + 1)];
    	Vector3d camKeyBas[(WINDOW_SIZE + 1)];
    	Vector3d camKeyBgs[(WINDOW_SIZE + 1)];

};