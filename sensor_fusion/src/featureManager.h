#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include "utility.h"

// #include "camodocal/camera_models/CameraFactory.h"
// #include "camodocal/camera_models/CataCamera.h"
// #include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace Eigen;

class FeaturePerId
{
    public:

        FeaturePerId(int _id, int _startFrame, double _startTime) :
            featureId(_id), startFrame(_startFrame), startTime(_startTime),
            usedNum(0), solveFlag(0), isDepth(0)
        {
        }
        const int featureId;
        double startTime;
        int startFrame;
        vector<sensor_fusion::cloud_info> featurePerFrame;
        int usedNum;
        int solveFlag;
        bool isDepth;
    
        int endFrame();
};

class FeatureManager : public ParamServer
{
private:
    mutex featLock;
    mutex pointLock;
    mutex imgLock;
    mutex odomLock;
    mutex imuLock;
    mutex lidarLock;

    deque<sensor_fusion::cloud_info> tempQueue;
    deque<pair<double, sensor_msgs::Image>> imgQueue; 
    deque<nav_msgs::Odometry> odomQueue;
    deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subCloudNearby;
    ros::Publisher  pubImgProjected;
    // ros::Publisher  pubPoint;
    ros::Publisher  pubFeature;
    ros::Publisher  pubExtractedCloudCam;

    sensor_msgs::PointCloud pointFeature;
    sensor_fusion::cloud_info cloudInfo;
    sensor_fusion::cloud_info pointInfo;
    sensor_fusion::cloud_info featureInfo; // hopefully create new message name

    pcl::PointCloud<PointType>::Ptr currentCloud;
    pcl::PointCloud<PointType>::Ptr laserCloudNearby;

    // vector<camodocal::CameraPtr> m_camera;
    double timeScanStart;
    double timeCloudInfoCur;
    double timeSent;

    Eigen::Affine3f transWorld2Cam;

    // feature-related
    float latestX;
    float latestY;
    float latestZ;
    float latestRoll;
    float latestPitch;
    float latestYaw;
    float latestImuRoll;
    float latestImuPitch;
    float latestImuYaw;
    double timeImageCur;
    bool odomAvailable;
    bool imuAvailable;

public: 

    int frameCount;
    int keyframeCount;
    list<FeaturePerId> pointFeaturesPerId;
    deque<pair<int, sensor_fusion::cloud_info>> featureQueue;

    FeatureManager();

    void allocateMemory();
  
    void resetParameters();

    void inputCloudNearby(const pcl::PointCloud<PointType>::Ptr cloudNearby);
    void inputIMU(const sensor_msgs::Imu imu);
    void inputPointFeature(const sensor_msgs::PointCloud point);
    void inputCloudInfo(const sensor_fusion::cloud_info cloudInfo);
    void inputOdom (const nav_msgs::Odometry odom);
    void inputImage (const sensor_msgs::Image img);

    void processFeature();
    void managePointFeature();
    void manageCloudFeature();
    void addKeyframe();
    void addKeyframe2();
    void associatePointFeature();
    void removeOldFeatures(double timestamp);
    bool addPointFeature();
    float compensatedParallax2(const FeaturePerId &it_per_id);
    void updateImuRPY();
    void updateOdometry();
    bool updateInitialPose();

    void visualizeAssociatedPoints(const sensor_msgs::ChannelFloat32 depths, const pcl::PointCloud<PointType>::Ptr localCloud);
};

#endif