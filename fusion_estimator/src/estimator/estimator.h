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
		void inputIMU(const sensor_msgs::ImuConstPtr &imu_msg);
		void inputImage(double t, const cv::Mat &img);
		void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg);

		// Imu-related
		bool IMUAvailable(const double t);
		bool getIMUInterval(double start, double end, vector<sensor_msgs::Imu> &imu_vec);

		// Image-related
		bool imageAvailable(const double time);
		bool getFirstImage(const double start, const double end, pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> &feature);

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
		void saveKeyframe();
		void updateLatestOdometry();
		void publishOdometry();

		// miscell
		Affine3f trans2Affine3f(Vector3d XYZ, Vector3d RPY);
		Pose3 trans2gtsamPose(Vector3d XYZ, Vector3d RPY);
		Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);

		deque<sensor_msgs::Imu> imuBuf;
		queue<sensor_msgs::PointCloud2> cloudBuf;
		queue<fusion_estimator::CloudInfo> cloudInfoBuf;
    	deque<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
		deque<nav_msgs::Odometry> odomBuf;
		fusion_estimator::CloudInfo cloudInfoIn; 

		mutex mBuf;
		mutex mProcess;
		mutex mPropagate;
		thread processThread;

		FeatureExtractor featureExtractor;
		FeatureTracker featureTracker;
		// MapOptimizer mapOptimizer;

		// Flags
		bool initThreadFlag;
		bool firstPointFlag;
		bool initCloudFlag;
		bool imageDeskewFlag;
		bool odomDeskewFlag;
		int deskewFlag;
    	cv::Mat rangeMat;

		// ros
		ros::Publisher pubCloud;
		ros::Publisher pubOdom;
		ros::Subscriber subOdom;

	    // gtsam
		int key;
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
		double scanStartTime;
		double scanEndTime;
		int cloudFrameCount;

		// image 
		int inputImageCnt;
		int imgFrameCount;
		double imgDeskewTime;
		double prevImgTime;
		double td;
		Vector3d g;
		Matrix3d ric[2];
		Vector3d tic[2];

		// imu
		double lastImuTime;
		double lastImuOptTime;
		int imuPointerCur;
		int imuPointerCam;
		
		Vector3d imuR[queueLength];
		Vector3d imuP[queueLength]; 
		double imuTime[queueLength];
		vector<sensor_msgs::Imu> initImuVector;

		gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
		gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
		gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
		gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
		gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
		gtsam::Vector noiseModelBetweenBias;
    	gtsam::PreintegratedImuMeasurements *imuIntegrator_;
		gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;

		// odom
	    float odomIncreX;
	    float odomIncreY;
	    float odomIncreZ;
        Eigen::Affine3f transStartInverse;
		Eigen::Affine3f transStart2Cam;
    	Eigen::Affine3f increOdomFront;
    	Eigen::Affine3f increOdomBack;
		double lastOdomTime;

        // map optimization
		std_msgs::Header cloudInfoInHeader;
        double cloudInfoInTime;

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
		gtsam::Pose3 prevPose_;
		gtsam::Vector3 prevVel_; 
		gtsam::NavState prevState_;
		gtsam::imuBias::ConstantBias prevBias_;
		gtsam::NavState prevOdom_;
		gtsam::imuBias::ConstantBias prevOdomBias_;
    	
    	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    	Vector3d camKeyPos[(WINDOW_SIZE + 1)];
    	Vector3d camKeyVel[(WINDOW_SIZE + 1)];
    	Matrix3d camKeyRot[(WINDOW_SIZE + 1)];
    	Vector3d camKeyBas[(WINDOW_SIZE + 1)];
    	Vector3d camKeyBgs[(WINDOW_SIZE + 1)];

};