#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include "utility.h"

// #include "camodocal/camera_models/CameraFactory.h"
// #include "camodocal/camera_models/CataCamera.h"
// #include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace Eigen;

class FeaturePerFrame
{
    public:
        FeaturePerFrame(const Eigen::Matrix<float, 6, 1> &_point, const Eigen::Matrix<float, 6, 1> &_initial, const double &_timestamp)
        {
            // measurements
            point.x() = _point(0);
            point.y() = _point(1);
            point.z() = 1.0;
            uv.x() = _point(2);
            uv.y() = _point(3);
            velocity.x() = _point(4);
            velocity.y() = _point(5);

            // initial pose
            initX = _initial(0);
            initY = _initial(1);
            initZ = _initial(2);
            initRoll = _initial(3);
            initPitch = _initial(4);
            initYaw = _initial(5);

            // time for frame
            timestamp = _timestamp;
            estimatedDepth = -1.0;
        }

        Vector3f point;
        Vector2f uv;
        Vector2f velocity;
        Vector3f point3d;
        float initX;
        float initY;
        float initZ;
        float initRoll;
        float initPitch;
        float initYaw;
        float estimatedDepth;
        double timestamp;
};

class FeaturePerId
{
    public:
        const int featureId;
        double startTime;
        int startFrame;
        vector<FeaturePerFrame> featurePerFrame;
        int usedNum;
        int solveFlag;
        bool isDepth;
        FeaturePerId(int _id, int _startFrame, double _startTime) :
            featureId(_id), startFrame(_startFrame), startTime(_startTime),
            usedNum(0), solveFlag(0), isDepth(0)
        {
        }
    
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
    double timeImageCur;
    double timeScanStart;
    double timeCloudInfoCur;
    double timeSent;

    Eigen::Affine3f transWorld2Cam;


public: 

    int frameCount;
    int keyframeCount;
    list<FeaturePerId> pointFeaturesPerId;
    deque<pair<int, FeaturePerFrame>> pointQueue;
    deque<sensor_fusion::cloud_info> featureQueue;

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