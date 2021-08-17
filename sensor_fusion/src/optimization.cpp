#include "utility.h"

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
// #include <gtsam/slam/GeneralSFMFactor.h>
// #include <gtsam/slam/dataset.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include "IncrementalFixedLagSmoother.h"
#include "FixedLagSmoother.h"
#include "featureManager.h"

// launch-prefix="xterm -e gdb --args"


using namespace std;
using namespace gtsam;
using namespace cv;
using symbol_shorthand::C;
using symbol_shorthand::P;

typedef SmartProjectionPoseFactor<Cal3_S2> SmartFactor;


class Optimization : public ParamServer
{
public:
    // GTSAM
    NonlinearFactorGraph factorGraph;
    Values initialEstimate;
    ISAM2 *isam;
    IncrementalFixedLagSmoother2 *smootherISAM2;

    ros::Subscriber subPoint;       
    ros::Subscriber subCloudInfo;    
    ros::Subscriber subImage;  
    ros::Subscriber subImu;
    ros::Subscriber subOdom;
    ros::Publisher  pubPoint;
    ros::Publisher  pubTestImage;
    ros::Publisher  pubCloudNearby;
    ros::Publisher  pubPath;
    ros::Publisher  pubInitialPath;
    ros::Publisher  pubOdomIncremental;
    int count;

    noiseModel::Diagonal::shared_ptr measNoise;
    noiseModel::Diagonal::shared_ptr poseNoise;
    noiseModel::Diagonal::shared_ptr pointNoise;
    noiseModel::Diagonal::shared_ptr odomNoise;

    map<int, int> numSeen;
    map<int, float> lastTime;
    map<int, bool> initialized;
    map<int, noiseModel::Gaussian::shared_ptr> marginalCovariances;
    map<Key, double> timestamps;
    map<int, SmartFactor::shared_ptr> smartPerId;

    std::mutex mtx;
    std::mutex featLock;

    // float type
    float latestX;
    float latestY;
    float latestZ;
    float latestRoll;
    float latestPitch;
    float latestYaw;

    int featureId;
    sensor_fusion::cloud_info cloudInfo;
    sensor_fusion::cloud_info featureInfo;
    ros::Time cloudInfoTimeIn;
    ros::Time timeFeatureIn;

    bool firstCloud;

    Eigen::Affine3f affineInitial;
    Eigen::Affine3f lastAffine;

    vector<pcl::PointCloud<PointType>::Ptr> deskewCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr keyPoses;

    pcl::PointCloud<PointType>::Ptr laserCloudDeskewLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudFeatureMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    map<int, pcl::PointCloud<PointType>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; 

    nav_msgs::Path globalPath;
    nav_msgs::Path initialPath;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    FeatureManager fManager;
    double timeLastIn;
    int numAssociatedPoint;

    Optimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        smootherISAM2 = new gtsam::IncrementalFixedLagSmoother2(timeLag, parameters);

        // Noises (TO-DO pose noise based on the params)
        // poseNoise = noiseModel::Isotropic::Sigma(9, 0.1); // 9 d.o.f
        measNoise   = noiseModel::Isotropic::Sigma(2, 1.0);// 1px in u and v
        poseNoise   = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        pointNoise  = noiseModel::Isotropic::Sigma(3, 0.2); // 3 d.o.f
        odomNoise   = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

        subPoint            = nh.subscribe<sensor_msgs::PointCloud>("/fusion/visual/point", 1000, &Optimization::pointHandler, this, ros::TransportHints().tcpNoDelay());
        subImage            = nh.subscribe<sensor_msgs::Image>(imgTopic, 1000, &Optimization::imgHandler, this, ros::TransportHints().tcpNoDelay());
        subImu              = nh.subscribe<sensor_msgs::Imu>(imuTopic, 1000, &Optimization::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom             = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 1000, &Optimization::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subCloudInfo        = nh.subscribe<sensor_fusion::cloud_info>("/fusion/feature/cloud_info", 1000, &Optimization::cloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        pubPoint            = nh.advertise<visualization_msgs::MarkerArray>("/fusion/feature/point_landmark", 1);
        pubCloudNearby      = nh.advertise<sensor_msgs::PointCloud2>("/fusion/mapping/cloud_nearby", 1);
        pubTestImage        = nh.advertise<sensor_msgs::Image>("fusion/feature/test_image", 1);
        pubPath             = nh.advertise<nav_msgs::Path>("fusion/mapping/path", 1);
        pubInitialPath      = nh.advertise<nav_msgs::Path>("fusion/mapping/initial_path", 1);
        pubOdomIncremental  = nh.advertise<nav_msgs::Odometry>("fusion/mapping/odometry_incremental", 1);
        
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        allocateMemory();

    }

    void allocateMemory()
    {
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        keyPoses.reset(new pcl::PointCloud<PointType>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudDeskewLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * HORIZON_SCAN);
        coeffSelCornerVec.resize(N_SCAN * HORIZON_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * HORIZON_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * HORIZON_SCAN);
        coeffSelSurfVec.resize(N_SCAN * HORIZON_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * HORIZON_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());


        firstCloud = false;

        // initialization
        count = 0;
        latestX = 0.0;
        latestY = 0.0;
        latestZ = 0.0;
        latestRoll = 0.0;
        latestPitch = 0.0;
        latestYaw = 0.0;

        timeLastIn = 0.0;
    }

    void updateInitialGuess()
    {
        if (timeFeatureIn.toSec() == timeLastIn)
        {
            printf("Already updated... cur: %f, last: %f\n", timeFeatureIn.toSec(), timeLastIn);
            return;
        }

        numAssociatedPoint = 0;

        affineInitial = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
        
        static Eigen::Affine3f lastImuAffine;
        // initialization
        if (cloudKeyPoses6D->points.empty())
        {
            latestRoll = featureInfo.imuRollInit;
            latestPitch = featureInfo.imuPitchInit;
            latestYaw = featureInfo.imuYawInit;
            lastImuAffine = pcl::getTransformation(0.0, 0.0, 0.0, featureInfo.imuRollInit, featureInfo.imuPitchInit, featureInfo.imuYawInit);
            printf("Initial: %f %f %f %f %f %f\n", 0, latestY, latestZ, latestRoll, latestPitch, latestYaw);
            return; 
        }
        static bool lastImuPreAffineAvailable = false;
        static bool skipped = false;
        static Eigen::Affine3f lastImuPreAffine;

        if (featureInfo.odomAvailable == true)
        {
            Eigen::Affine3f affineBack = pcl::getTransformation(featureInfo.initialGuessX, featureInfo.initialGuessY, featureInfo.initialGuessZ,
                                                featureInfo.initialGuessRoll, featureInfo.initialGuessPitch, featureInfo.initialGuessYaw);  
            printf("Initial odom: %f %f %f %f %f %f\n", featureInfo.initialGuessX, featureInfo.initialGuessY, featureInfo.initialGuessZ, featureInfo.initialGuessRoll, featureInfo.initialGuessPitch, featureInfo.initialGuessYaw);
            if (!skipped)
            {
                printf("Skipped\n");
                skipped = true;
                return;
            }
            if (!lastImuPreAffineAvailable)
            {
                lastImuPreAffine = affineBack;
                lastImuPreAffineAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreAffine.inverse() * affineBack;
                Eigen::Affine3f transTobe = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
                printf("Odom: %f %f %f %f %f %f\n", latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
                lastImuPreAffine = affineBack;
                lastImuAffine = pcl::getTransformation(0, 0, 0, featureInfo.imuRollInit, featureInfo.imuPitchInit, featureInfo.imuYawInit);
                return;
            }
        }

        if (featureInfo.imuAvailable == true)
        {
            printf("Initial rpy: %f %f %f\n", featureInfo.imuRollInit, featureInfo.imuPitchInit, featureInfo.imuYawInit);
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, featureInfo.imuRollInit, featureInfo.imuPitchInit, featureInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuAffine.inverse() * transBack;
            Eigen::Affine3f transTobe = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
            printf("RPY: %f %f %f %f %f %f\n", latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
            lastImuAffine = pcl::getTransformation(0, 0, 0, featureInfo.imuRollInit, featureInfo.imuPitchInit, featureInfo.imuYawInit);
            return;
        }
    }

    void addOdomFactor()
    {
        Pose3 poseTo = float2gtsamPose(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
        printf("addOdomFactor's key count: %d\n", cloudKeyPoses6D->points.size());
        if (cloudKeyPoses6D->points.empty())
        {
            factorGraph.add(PriorFactor<Pose3>(Symbol('x', 0), poseTo, poseNoise));
            timestamps[Symbol('x', 0)] = timeFeatureIn.toSec();
            initialEstimate.insert(Symbol('x', 0), poseTo);
        } else {
            if (cloudKeyPoses6D->points.size() == 1)
                factorGraph.add(PriorFactor<Pose3>(Symbol('x', 1), poseTo, poseNoise));

            Pose3 poseFrom = pclPoint2gtsamPose(cloudKeyPoses6D->points.back());
            int keyFrom = cloudKeyPoses6D->size() - 1;
            int keyTo = cloudKeyPoses6D->size();
            factorGraph.add(BetweenFactor<Pose3>(Symbol('x', keyFrom), Symbol('x', keyTo), 
                                                poseFrom.between(poseTo), odomNoise));
            initialEstimate.insert(Symbol('x', keyTo), poseTo);
            timestamps[Symbol('x', keyTo)] = timeFeatureIn.toSec();
        }
    }


    void addPointFactor2()
    {
        if (featureId < 0)
            return;

        TicToc tictoc;

        visualization_msgs::MarkerArray pointLandmark;
        
        printf("addPointFactor: %d -- cur: %f, last: %f, est: %f\n", featureId, timeFeatureIn.toSec(), timeLastIn, featureInfo.estimatedDepth);

        Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));
        Point2 measurement = Point2(featureInfo.uv.x, featureInfo.uv.y);
        Eigen::Affine3f curPose = pcl::getTransformation(featureInfo.initialGuessX,
                                                        featureInfo.initialGuessY,
                                                        featureInfo.initialGuessZ,
                                                        featureInfo.initialGuessRoll,
                                                        featureInfo.initialGuessPitch,
                                                        featureInfo.initialGuessYaw); 

        // update counter
        if (numSeen.find(featureId) == numSeen.end())
        {
            numSeen[featureId] = 1;
        } else {
            numSeen[featureId]++;
        }

        if (featureInfo.estimatedDepth >= 0.0)
        {
            // convert point to world frame
            float worldX = curPose(0,0)*featureInfo.point3d.x+curPose(0,1)*featureInfo.point3d.y+curPose(0,2)*featureInfo.point3d.z+curPose(0,3);
            float worldY = curPose(1,0)*featureInfo.point3d.x+curPose(1,1)*featureInfo.point3d.y+curPose(1,2)*featureInfo.point3d.z+curPose(1,3);
            float worldZ = curPose(2,0)*featureInfo.point3d.x+curPose(2,1)*featureInfo.point3d.y+curPose(2,2)*featureInfo.point3d.z+curPose(2,3);
            Point3 worldPoint = Point3(worldX, worldY, worldZ);

            factorGraph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
                measurement, measNoise, Symbol('x', cloudKeyPoses6D->size()), Symbol('l', featureId), K);
            timestamps[Symbol('l', featureId)] = timeFeatureIn.toSec();

            // visualization (off for operation)
            visualization_msgs::Marker marker;
            marker.action = 0;
            marker.type = 2; // sphere
            marker.ns = "points";
            marker.id = featureId;
            marker.pose.position.x = worldPoint.x();
            marker.pose.position.y = worldPoint.y();
            marker.pose.position.z = worldPoint.z();
            marker.header.frame_id = mapFrame;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;
            marker.color.r = 0.5451;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;
            pointLandmark.markers.push_back(marker);
            numAssociatedPoint++;

            if (!initialized[featureId])
            {
                initialized[featureId] = true;
                initialEstimate.insert(Symbol('l', featureId), worldPoint);
                if (marginalCovariances.find(featureId) == marginalCovariances.end()) // first initialized
                    factorGraph.add(PriorFactor<Point3>(Symbol('l', featureId), worldPoint, pointNoise));
                else // marginalized before
                    factorGraph.add(PriorFactor<Point3>(Symbol('l', featureId), worldPoint, marginalCovariances[featureId]));
            }
        }
        else
        {
            if (!initialized[featureId])
            {
                if (numSeen[featureId] >= 10)
                {
                    SmartFactor::shared_ptr smartfactor = smartPerId[featureId];
                    factorGraph.push_back(smartfactor);
                }

                if (smartPerId.find(featureId) == smartPerId.end()) // first initialized
                {
                    SmartFactor::shared_ptr smartfactor(new SmartFactor(measNoise, K));
                    Point2 measurement = Point2(featureInfo.uv.x, featureInfo.uv.y);
                    smartfactor->add(measurement, Symbol('x', (int)cloudKeyPoses6D->size()));
                    smartPerId[featureId] = smartfactor;
                }
                else
                {
                    SmartFactor::shared_ptr smartfactor = smartPerId[featureId];
                    Point2 measurement = Point2(featureInfo.uv.x, featureInfo.uv.y);
                    smartfactor->add(measurement, Symbol('x', (int)cloudKeyPoses6D->size()));
                }                
            }
        }
            
        printf("# of points: %d\n", numAssociatedPoint);
        pubPoint.publish(pointLandmark);
        ROS_WARN("Add point factor time: %fms\n", tictoc.toc());

    }

    void addPointFactor()
    {
        if (featureId < 0)
            return;

        TicToc tictoc;

        int numPoint = 0;

        visualization_msgs::MarkerArray pointLandmark;
        
        Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));

        printf("addPointFactor: %d -- cur: %f, last: %f, est: %f\n", featureId, timeFeatureIn.toSec(), timeLastIn, featureInfo.estimatedDepth);

        Point2 measurement = Point2(featureInfo.uv.x, featureInfo.uv.y);
        Eigen::Affine3f curPose = pcl::getTransformation(featureInfo.initialGuessX,
                                                        featureInfo.initialGuessY,
                                                        featureInfo.initialGuessZ,
                                                        featureInfo.initialGuessRoll,
                                                        featureInfo.initialGuessPitch,
                                                        featureInfo.initialGuessYaw); 
        // update counter
        if (numSeen.find(featureId) == numSeen.end())
        {
            numSeen[featureId] = 1;
        } else {
            numSeen[featureId]++;
        }

        // create different factors for different points
        if (featureInfo.estimatedDepth >= 0.0)
        {
            // convert point to world frame
            float worldX = curPose(0,0)*featureInfo.point3d.x+curPose(0,1)*featureInfo.point3d.y+curPose(0,2)*featureInfo.point3d.z+curPose(0,3);
            float worldY = curPose(1,0)*featureInfo.point3d.x+curPose(1,1)*featureInfo.point3d.y+curPose(1,2)*featureInfo.point3d.z+curPose(1,3);
            float worldZ = curPose(2,0)*featureInfo.point3d.x+curPose(2,1)*featureInfo.point3d.y+curPose(2,2)*featureInfo.point3d.z+curPose(2,3);
            Point3 worldPoint = Point3(worldX, worldY, worldZ);

            factorGraph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
                measurement, measNoise, Symbol('x', cloudKeyPoses6D->size()), Symbol('l', featureId), K);
            timestamps[Symbol('l', featureId)] = timeFeatureIn.toSec();

            // visualization (off for operation)
            visualization_msgs::Marker marker;
            marker.action = 0;
            marker.type = 2; // sphere
            marker.ns = "points";
            marker.id = featureId;
            marker.pose.position.x = worldPoint.x();
            marker.pose.position.y = worldPoint.y();
            marker.pose.position.z = worldPoint.z();
            marker.header.frame_id = mapFrame;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;
            marker.color.r = 0.5451;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;
            pointLandmark.markers.push_back(marker);

            if (!initialized[featureId])
            {
                initialized[featureId] = true;
                initialEstimate.insert(Symbol('l', featureId), worldPoint);
                if (marginalCovariances.find(featureId) == marginalCovariances.end()) // first initialized
                    factorGraph.add(PriorFactor<Point3>(Symbol('l', featureId), worldPoint, pointNoise));
                else // marginalized before
                    factorGraph.add(PriorFactor<Point3>(Symbol('l', featureId), worldPoint, marginalCovariances[featureId]));
            }
        }
        else
        {
            printf("Not associated point\n");
            SmartFactor::shared_ptr smartfactor(new SmartFactor(measNoise, K));
            Point2 measurement = Point2(featureInfo.uv.x, featureInfo.uv.y);
            smartfactor->add(measurement, Symbol('x', (int)cloudKeyPoses6D->size()));
            factorGraph.push_back(smartfactor);
        }
        numPoint++;
            
        printf("# of points: %d\n", numPoint);
        pubPoint.publish(pointLandmark);
        ROS_WARN("Add point factor time: %fms\n", tictoc.toc());

    }

    // void addPointFactor2()
    // {
    //     TicToc tictoc;

    //     if (cloudInfo.point_feature.points.size() == 0) 
    //         return;

    //     int numPoint = 0;
    //     visualization_msgs::MarkerArray pointLandmark;
    //     // cv::Mat testImg(ROW, COL, CV_8UC3, Scalar(255, 255, 255)); 
    //     // cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

    //     // cv_ptr->encoding = "bgr8";
    //     // cv_ptr->header.stamp = cloudInfo.header.stamp;
    //     // cv_ptr->header.frame_id = "image";
    //     // cv_ptr->image = testImg;

    //     Eigen::Affine3f affineNow = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
    //     float camPosX, camPosY, camPosZ, camRotX, camRotY, camRotZ;
    //     pcl::getTranslationAndEulerAngles(affineNow, camPosX, camPosY, camPosZ, camRotX, camRotY, camRotZ);

    //     Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));

    //     for (size_t i = 0; i < cloudInfo.point_feature.points.size(); ++i)
    //     {
    //         Point2 measurement = Point2(double(cloudInfo.point_feature.channels[4].values[i]), 
    //                                     double(cloudInfo.point_feature.channels[5].values[i]));
            
    //         // TO-DO: modify the method to check association
    //         bool associated = !(cloudInfo.point_feature.channels[1].values[i] == 0.0
    //                             && cloudInfo.point_feature.channels[2].values[i] == 0.0
    //                             && cloudInfo.point_feature.channels[3].values[i] == 0.0);
            
    //         int id = int(cloudInfo.point_feature.channels[0].values[i]);

    //         if (numSeen.find(id) == numSeen.end())
    //             numSeen[id] = 1;
    //         else
    //             numSeen[id]++;

    //         if (associated)
    //         {
    //             if (numSeen[id] >= 1)
    //             {
    //                 // printf("id %d Num seen: %d\n", id, numSeen[id]);
    //                 lastTime[id] = cloudInfoTimeIn.toSec();

    //                 // convert point to world frame
    //                 float localX = cloudInfo.point_feature.channels[1].values[i];
    //                 float localY = cloudInfo.point_feature.channels[2].values[i];
    //                 float localZ = cloudInfo.point_feature.channels[3].values[i];
    //                 float worldX = affineNow(0,0)*localX+affineNow(0,1)*localY+affineNow(0,2)*localZ+affineNow(0,3);
    //                 float worldY = affineNow(1,0)*localX+affineNow(1,1)*localY+affineNow(1,2)*localZ+affineNow(1,3);
    //                 float worldZ = affineNow(2,0)*localX+affineNow(2,1)*localY+affineNow(2,2)*localZ+affineNow(2,3);
    //                 Point3 worldPoint = Point3(worldX, worldY, worldZ);

    //                 // project point for checking
    //                 Pose3 camPose = float2gtsamPose(camPosX, camPosY, camPosZ, camRotX, camRotY, camRotZ);
    //                 PinholeCamera<Cal3_S2> camera(camPose, *K);
    //                 Point2 estimation = camera.project(worldPoint);

    //                 // add projection factor 
    //                 float reprojError = sqrt((measurement.x()-estimation.x())*(measurement.x()-estimation.x()) +
    //                                             (measurement.y()-estimation.y())*(measurement.y()-estimation.y()));

    //                 if (reprojError < projectionErrorThres)
    //                 {
    //                     factorGraph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
    //                         measurement, measNoise, Symbol('x', cloudKeyPoses6D->size()), Symbol('l', id), K);
    //                     timestamps[Symbol('l', id)] = cloudInfoTimeIn.toSec();

    //                     // visualization (off for operation)
    //                     visualization_msgs::Marker marker;
    //                     marker.action = 0;
    //                     marker.type = 2; // sphere
    //                     marker.ns = "points";
    //                     marker.id = id;
    //                     marker.pose.position.x = worldPoint.x();
    //                     marker.pose.position.y = worldPoint.y();
    //                     marker.pose.position.z = worldPoint.z();
    //                     marker.header.frame_id = mapFrame;
    //                     marker.scale.x = 0.2;
    //                     marker.scale.y = 0.2;
    //                     marker.scale.z = 0.2;
    //                     marker.color.r = 0.5451;
    //                     marker.color.g = 0.0;
    //                     marker.color.b = 1.0;
    //                     marker.color.a = 1.0;
    //                     pointLandmark.markers.push_back(marker);

    //                     if (!initialized[id])
    //                     {
    //                         initialized[id] = true;
    //                         initialEstimate.insert(Symbol('l', id), worldPoint);
    //                         if (marginalCovariances.find(id) == marginalCovariances.end()) // first initialized
    //                             factorGraph.add(PriorFactor<Point3>(Symbol('l', id), worldPoint, pointNoise));
    //                         else // marginalized before
    //                             factorGraph.add(PriorFactor<Point3>(Symbol('l', id), worldPoint, marginalCovariances[id]));
    //                     }
    //                     // cv::putText(cv_ptr->image, to_string(id), Point(measurement.x(), measurement.y()), 
    //                     //             cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 0.5, cv::LINE_AA);
    //                 }
    //                 else
    //                 {
    //                     // cv::putText(cv_ptr->image, to_string(id), Point(measurement.x(), measurement.y()), 
    //                     //             cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 255), 0.5, cv::LINE_AA);
    //                 }
                    
    //                 // // visualization (test image for projection)
    //                 // cv::line(cv_ptr->image, cv::Point(measurement.x(), measurement.y()), cv::Point(estimation.x(), estimation.y()), Scalar(255, 0, 0), 1, LINE_8);

    //                 numPoint++;
    //             }
            
    //         }
    //         else
    //         {
    //             if (numSeen[id] >= 5)
    //             {

    //             }
    //         }
    //     }


    //     printf("# of points: %d\n", numPoint);
    //     pubPoint.publish(pointLandmark);
    //     // pubTestImage.publish(cv_ptr->toImageMsg());

    //     ROS_WARN("Add point factor time: %fms\n", tictoc.toc());
    // }

    void updatePath(const PointTypePose& pose_in, nav_msgs::Path& path)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        // // euigon
        // geometry_msgs::PoseStamped gps_posestamped;
        // transformPose(pose_stamped, gps_posestamped);
        // gpsGlobalPath.poses.push_back(gps_posestamped);
        path.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        if (timeFeatureIn.toSec() == timeLastIn)
            return;
        
        nav_msgs::Odometry odomIncre;
        Eigen::Affine3f affineFinal = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw) * affineL2C.inverse();
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(affineFinal, x, y, z, roll, pitch, yaw);
        odomIncre.header.stamp = timeFeatureIn;
        odomIncre.header.frame_id = odometryFrame;
        odomIncre.child_frame_id = "odom_mapping";
        odomIncre.pose.pose.position.x = x;
        odomIncre.pose.pose.position.y = y;
        odomIncre.pose.pose.position.z = z;
        odomIncre.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubOdomIncremental.publish(odomIncre);
        ROS_WARN("optimization.cpp: publish odometry incremental");
        printf("x: %f, y: %f, z: %f, r: %f, p: %f, y: %f\n", x, y, z, roll, pitch, yaw);
        // TO-DO: consider loop closing and publish relative pose
    }  

    void publishFrames()
    {
        if (cloudKeyPoses6D->points.empty())
            return;

        fManager.inputCloudNearby(laserCloudFromMap);
        publishCloud(&pubCloudNearby, laserCloudFromMap, timeFeatureIn, mapFrame);

        // publish key poses
        static tf::TransformBroadcaster tfMap2Cam;
        tf::Transform map2Cam = tf::Transform(tf::createQuaternionFromRPY(latestRoll, latestPitch, latestYaw), 
                                              tf::Vector3(latestX, latestY, latestZ));
        tfMap2Cam.sendTransform(tf::StampedTransform(map2Cam, timeFeatureIn, mapFrame, camFrame));
        
        ROS_WARN("publish frames");

        // publish global path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeFeatureIn;
            globalPath.header.frame_id = mapFrame;
            pubPath.publish(globalPath);
        }

        // temp
        if (pubInitialPath.getNumSubscribers() != 0)
        {
            initialPath.header.stamp = timeFeatureIn;
            initialPath.header.frame_id = mapFrame;
            pubInitialPath.publish(initialPath);
        }

    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        fManager.inputIMU(thisImu);
    }

    void pointHandler(const sensor_msgs::PointCloudConstPtr &msg)
    {
        fManager.inputPointFeature(*msg);
    }
        
    void cloudInfoHandler(const sensor_fusion::cloud_infoConstPtr &msgIn)
    {
        fManager.inputCloudInfo(*msgIn);
    }

    void odometryHandler(const nav_msgs::OdometryConstPtr &odomMsg)
    {
        fManager.inputOdom(*odomMsg);
    }

    void imgHandler(const sensor_msgs::ImageConstPtr &imgMsg)
    {
        fManager.inputImage(*imgMsg);
    }

    void processOptimization()
    {
        ros::Rate rate(500);

        while (ros::ok())
        {
            rate.sleep();

            mtx.lock();

            if (!fManager.featureQueue.empty())
            {
                featureId = fManager.featureQueue.front().first;
                featureInfo = fManager.featureQueue.front().second;
                timeFeatureIn = featureInfo.header.stamp;

                featLock.lock();
                fManager.featureQueue.pop_front();
                featLock.unlock();

                ROS_WARN("id: %d, featureInfo: %f, Last: %f\n", featureId, timeFeatureIn.toSec(), timeLastIn);

                if (timeFeatureIn.toSec() >= timeLastIn)
                {
                    if (featureId < 0) // pointcloud feature
                    { 
                        laserCloudDeskewLast->clear();
                        pcl::fromROSMsg(featureInfo.cloud_deskewed, *laserCloudDeskewLast);
                        pcl::fromROSMsg(featureInfo.cloud_corner, *laserCloudCornerLast);
                        pcl::fromROSMsg(featureInfo.cloud_surface, *laserCloudSurfLast);
                        printf("CloudDeskewSize: %d\n", (int)laserCloudDeskewLast->size());
                        
                        downsampleCurrentScan();

                        // scan2MapOptimization(); 
                    }

                    updateInitialGuess();

                    optimize();

                    extractNearby();

                    publishOdometry();

                    publishFrames();

                    timeLastIn = timeFeatureIn.toSec();

                }
                ROS_WARN("Keyframe count: %d\n", count);

            } 

            fManager.removeOldFeatures(timeLastIn - timeLag);       

            mtx.unlock();
        }
    }

    // void featureInfoHandler(const sensor_fusion::cloud_infoConstPtr &cloudInfoMsg)
    // {
    //     laserCloudDeskewLast->clear();

    //     cloudInfo = *cloudInfoMsg;
    //     cloudInfoTimeIn = cloudInfo.header.stamp;

    //     pcl::fromROSMsg(cloudInfo.cloud_deskewed, *laserCloudDeskewLast);
    //     pcl::fromROSMsg(cloudInfo.cloud_corner, *laserCloudCornerLast);
    //     pcl::fromROSMsg(cloudInfo.cloud_surface, *laserCloudSurfLast);

    //     if (cloudInfo.point_feature.points.size() == 0)
    //     {
    //         ROS_WARN("Cloud Handler");
    //         firstCloud = true;
    //     } else {
    //         ROS_WARN("Point Handler");
    //     }

    //     if (!firstCloud)
    //     {
    //         return;
    //     }

    //     // TO-DO: Here if the info is from point, we have to fill in the WINDOW_SIZE
    //     // For now, skip everything before the first cloud Handler

    //     lock_guard<mutex> lock(mtx);

    //     Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));

    //     static double timeLastProcessing = -1;

    //     if (cloudInfoTimeIn.toSec() - timeLastProcessing >= mappingProcessInterval)
    //     {
    //         timeLastProcessing = cloudInfoTimeIn.toSec();
            
    //         updateInitialGuess();

    //         downsampleCurrentScan();

    //         // scan2MapOptimization(); 

    //         optimize();

    //         extractNearby();

    //         publishOdometry();

    //         publishFrames();

    //         ROS_WARN("Keyframe count: %d\n", count);
    //     }
    // }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast); 
        downSizeFilterCorner.filter(*laserCloudCornerLastDS); // Down-sampled corner points
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS); // Down-sampled surface points
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    // void updatePointAssociateToMap()
    // {
    //     transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    // }

    // void cornerOptimization()
    // {
    //     updatePointAssociateToMap();

    //     #pragma omp parallel for num_threads(numberOfCores)
    //     for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    //     {
    //         PointType pointOri, pointSel, coeff;
    //         std::vector<int> pointSearchInd;
    //         std::vector<float> pointSearchSqDis;

    //         pointOri = laserCloudCornerLastDS->points[i];
    //         pointAssociateToMap(&pointOri, &pointSel); // pointSel = (k+1)th odom * pointOri (global로 transform)
    //         kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 
    //         // k까지의 CornerMap에서 k+1 odom에서의 corner point와 가장 가까운 점 찾기

    //         cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    //         cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    //         cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
    //         if (pointSearchSqDis[4] < 1.0) { // Furtherest of all should be within 1.0m
    //             float cx = 0, cy = 0, cz = 0;
    //             for (int j = 0; j < 5; j++) {
    //                 cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
    //                 cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
    //                 cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
    //             }
    //             cx /= 5; cy /= 5;  cz /= 5; // nearest 5's average position

    //             float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    //             for (int j = 0; j < 5; j++) {
    //                 float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
    //                 float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
    //                 float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

    //                 a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    //                 a22 += ay * ay; a23 += ay * az;
    //                 a33 += az * az;
    //             }
    //             a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;
    //             // average cross product ? 

    //             matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
    //             matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
    //             matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

    //             cv::eigen(matA1, matD1, matV1); // matD1: eigenvalues, matV1: eigenvectors

    //             if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) { // one large and two small eigenvalues represent an edge line segment

    //                 float x0 = pointSel.x;
    //                 float y0 = pointSel.y;
    //                 float z0 = pointSel.z;
    //                 float x1 = cx + 0.1 * matV1.at<float>(0, 0);
    //                 float y1 = cy + 0.1 * matV1.at<float>(0, 1);
    //                 float z1 = cz + 0.1 * matV1.at<float>(0, 2);
    //                 float x2 = cx - 0.1 * matV1.at<float>(0, 0);
    //                 float y2 = cy - 0.1 * matV1.at<float>(0, 1);
    //                 float z2 = cz - 0.1 * matV1.at<float>(0, 2);

    //                 float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
    //                                 + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
    //                                 + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

    //                 float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

    //                 float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
    //                           + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

    //                 float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
    //                            - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    //                 float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
    //                            + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    //                 float ld2 = a012 / l12;

    //                 float s = 1 - 0.9 * fabs(ld2);

    //                 coeff.x = s * la;
    //                 coeff.y = s * lb;
    //                 coeff.z = s * lc;
    //                 coeff.intensity = s * ld2;

    //                 if (s > 0.1) {
    //                     laserCloudOriCornerVec[i] = pointOri; // local frame corner point
    //                     coeffSelCornerVec[i] = coeff; // coefficient for each corner
    //                     laserCloudOriCornerFlag[i] = true;
    //                 }
    //             }
    //         }
    //     }
    // }

    // void surfOptimization()
    // {
    //     updatePointAssociateToMap();

    //     #pragma omp parallel for num_threads(numberOfCores)
    //     for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    //     {
    //         PointType pointOri, pointSel, coeff;
    //         std::vector<int> pointSearchInd;
    //         std::vector<float> pointSearchSqDis;

    //         pointOri = laserCloudSurfLastDS->points[i];
    //         pointAssociateToMap(&pointOri, &pointSel); 
    //         kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

    //         Eigen::Matrix<float, 5, 3> matA0;
    //         Eigen::Matrix<float, 5, 1> matB0;
    //         Eigen::Vector3f matX0;

    //         matA0.setZero();
    //         matB0.fill(-1);
    //         matX0.setZero();

    //         if (pointSearchSqDis[4] < 1.0) {
    //             for (int j = 0; j < 5; j++) {
    //                 matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
    //                 matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
    //                 matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
    //             }

    //             matX0 = matA0.colPivHouseholderQr().solve(matB0);

    //             float pa = matX0(0, 0);
    //             float pb = matX0(1, 0);
    //             float pc = matX0(2, 0);
    //             float pd = 1;

    //             float ps = sqrt(pa * pa + pb * pb + pc * pc);
    //             pa /= ps; pb /= ps; pc /= ps; pd /= ps;

    //             bool planeValid = true;
    //             for (int j = 0; j < 5; j++) {
    //                 if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
    //                          pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
    //                          pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
    //                     planeValid = false;
    //                     break;
    //                 }
    //             }

    //             if (planeValid) {
    //                 float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

    //                 float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
    //                         + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

    //                 coeff.x = s * pa;
    //                 coeff.y = s * pb;
    //                 coeff.z = s * pc;
    //                 coeff.intensity = s * pd2;

    //                 if (s > 0.1) {
    //                     laserCloudOriSurfVec[i] = pointOri;
    //                     coeffSelSurfVec[i] = coeff;
    //                     laserCloudOriSurfFlag[i] = true;
    //                 }
    //             }
    //         }
    //     }
    // }

    // void combineOptimizationCoeffs()
    // {
    //     // combine corner coeffs
    //     for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
    //         if (laserCloudOriCornerFlag[i] == true){ // 특정 조건을 만족하는 corner point들만 push_back
    //             laserCloudOri->push_back(laserCloudOriCornerVec[i]);
    //             coeffSel->push_back(coeffSelCornerVec[i]);
    //         }
    //     }
    //     // combine surf coeffs
    //     for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
    //         if (laserCloudOriSurfFlag[i] == true){ // 특정 조건을 만족하는 surface point들만 Push_back
    //             laserCloudOri->push_back(laserCloudOriSurfVec[i]);
    //             coeffSel->push_back(coeffSelSurfVec[i]);
    //         }
    //     }
    //     // reset flag for next iteration
    //     std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    //     std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    // }

    // bool LMOptimization(int iterCount)
    // {
    //     // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    //     // lidar <- camera      ---     camera <- lidar
    //     // x = z                ---     x = y
    //     // y = x                ---     y = z
    //     // z = y                ---     z = x
    //     // roll = yaw           ---     roll = pitch
    //     // pitch = roll         ---     pitch = yaw
    //     // yaw = pitch          ---     yaw = roll

    //     // lidar -> camera
    //     float srx = sin(transformTobeMapped[1]);
    //     float crx = cos(transformTobeMapped[1]);
    //     float sry = sin(transformTobeMapped[2]);
    //     float cry = cos(transformTobeMapped[2]);
    //     float srz = sin(transformTobeMapped[0]);
    //     float crz = cos(transformTobeMapped[0]);

    //     int laserCloudSelNum = laserCloudOri->size();
    //     if (laserCloudSelNum < 50) {
    //         return false;
    //     }

    //     cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    //     cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    //     cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    //     cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    //     cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    //     cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    //     PointType pointOri, coeff;

    //     for (int i = 0; i < laserCloudSelNum; i++) {
    //         // lidar -> camera
    //         pointOri.x = laserCloudOri->points[i].y;
    //         pointOri.y = laserCloudOri->points[i].z;
    //         pointOri.z = laserCloudOri->points[i].x;
    //         // lidar -> camera
    //         coeff.x = coeffSel->points[i].y;
    //         coeff.y = coeffSel->points[i].z;
    //         coeff.z = coeffSel->points[i].x;
    //         coeff.intensity = coeffSel->points[i].intensity;
    //         // in camera
    //         float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
    //                   + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
    //                   + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

    //         float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
    //                   + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
    //                   + ((-cry*crz - srx*sry*srz)*pointOri.x 
    //                   + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

    //         float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
    //                   + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
    //                   + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
    //         // lidar -> camera
    //         matA.at<float>(i, 0) = arz;
    //         matA.at<float>(i, 1) = arx;
    //         matA.at<float>(i, 2) = ary;
    //         matA.at<float>(i, 3) = coeff.z;
    //         matA.at<float>(i, 4) = coeff.x;
    //         matA.at<float>(i, 5) = coeff.y;
    //         matB.at<float>(i, 0) = -coeff.intensity;
    //     }

    //     cv::transpose(matA, matAt);
    //     matAtA = matAt * matA;
    //     matAtB = matAt * matB;
    //     cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    //     if (iterCount == 0) {

    //         cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
    //         cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
    //         cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

    //         cv::eigen(matAtA, matE, matV);
    //         matV.copyTo(matV2);

    //         isDegenerate = false;
    //         float eignThre[6] = {100, 100, 100, 100, 100, 100};
    //         for (int i = 5; i >= 0; i--) {
    //             if (matE.at<float>(0, i) < eignThre[i]) {
    //                 for (int j = 0; j < 6; j++) {
    //                     matV2.at<float>(i, j) = 0;
    //                 }
    //                 isDegenerate = true;
    //             } else {
    //                 break;
    //             }
    //         }
    //         matP = matV.inv() * matV2;
    //     }

    //     if (isDegenerate)
    //     {
    //         cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
    //         matX.copyTo(matX2);
    //         matX = matP * matX2; // V^(-1) * V * delta(X)
    //     }

    //     // return solution_x after optimization
    //     transformTobeMapped[0] += matX.at<float>(0, 0);
    //     transformTobeMapped[1] += matX.at<float>(1, 0);
    //     transformTobeMapped[2] += matX.at<float>(2, 0);
    //     transformTobeMapped[3] += matX.at<float>(3, 0);
    //     transformTobeMapped[4] += matX.at<float>(4, 0);
    //     transformTobeMapped[5] += matX.at<float>(5, 0);

    //     float deltaR = sqrt(
    //                         pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
    //                         pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
    //                         pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    //     float deltaT = sqrt(
    //                         pow(matX.at<float>(3, 0) * 100, 2) +
    //                         pow(matX.at<float>(4, 0) * 100, 2) +
    //                         pow(matX.at<float>(5, 0) * 100, 2));

    //     if (deltaR < 0.05 && deltaT < 0.05) {
    //         return true; // converged
    //     }
    //     return false; // keep optimizing
    // }

    // void scan2MapOptimization()
    // {
    //     if (cloudKeyPoses3D->points.empty())
    //         return;

    //     if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    //     {
    //         kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS); // local corner map
    //         kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS); // local suf map 

    //         for (int iterCount = 0; iterCount < 30; iterCount++)
    //         {
    //             laserCloudOri->clear();
    //             coeffSel->clear();

    //             cornerOptimization();
    //             surfOptimization();

    //             combineOptimizationCoeffs();

    //             if (LMOptimization(iterCount) == true)
    //                 break;              
    //         }

    //         transformUpdate();
    //     } else {
    //         ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    //     }
    // }

    // void transformUpdate()
    // {

    //     // imu 값으로 얻은 위치와 transformToBeMapped 위치에 대해 interpolation을 한다. (roll, pitch 만)
    //     if (cloudInfo.imuAvailable == true)
    //     {
    //         if (std::abs(cloudInfo.imuPitchInit) < 1.4)
    //         {
    //             double imuWeight = imuRPYWeight;
    //             tf::Quaternion imuQuaternion;
    //             tf::Quaternion transformQuaternion;
    //             double rollMid, pitchMid, yawMid;

    //             // slerp roll
    //             transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
    //             imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
    //             tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid); 
    //             // slerp: spherical linear interpolation --> transformToBeMapped부터 imuRollInit까지 imuRPYWeight ratio로 slerp
    //             transformTobeMapped[0] = rollMid;

    //             // slerp pitch
    //             transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
    //             imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
    //             tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
    //             transformTobeMapped[1] = pitchMid;
    //         }
    //     }

    //     transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    //     transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    //     transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    //     incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped); // map optimized된 이후의 transformTobeMapped
    // }

    void extractNearby()
    {
        TicToc tictoc;

        if (cloudKeyPoses3D->points.empty() || featureId >= 0)
            return;
        
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesForStacking(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // 1. Extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree; input all the key poses 3D
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadiusSmall, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            if (i > surroundingKeyframeStackingThreshold)
                break;
            int id = pointSearchInd[i];
            surroundingKeyPosesForStacking->push_back(cloudKeyPoses3D->points[id]);
        }

        pointSearchInd.clear();
        pointSearchSqDis.clear();

        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadiusLarge, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        printf("surroundingKeyposes: %d\n", (int)surroundingKeyPoses->size());

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses); // radius 주변 pose들 또한 voxelize를 한다.
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // 2. Also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeFeatureIn.toSec() - cloudKeyPoses6D->points[i].time < 0.5)
                surroundingKeyPosesForStacking->push_back(cloudKeyPoses3D->points[i]);
            if (timeFeatureIn.toSec() - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 3. Extract cloud from nearby key poses
        laserCloudFromMap->clear();
        printf("surroundingKeyposesforstacking: %d\n", (int)surroundingKeyPosesForStacking->size());
        for (int i = 0; i < (int)surroundingKeyPosesForStacking->size(); ++i)
        {
            float distance = pointDistance(surroundingKeyPosesForStacking->points[i], cloudKeyPoses3D->back());
            printf("pointDistance: %f\n", distance);
            if (distance > surroundingKeyframeSearchRadiusSmall)
                continue;
            
            int thisKeyInd = (int) surroundingKeyPosesForStacking->points[i].intensity;
            // already have been converted to global frame
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
            {
                *laserCloudFromMap += laserCloudMapContainer[thisKeyInd];
                printf("Already in the global map\n");
            } else {
                pcl::PointCloud<PointType> globalCloud = *transformPointCloud(deskewCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudFromMap += globalCloud;
                printf("Newly added to the global map\n");
                laserCloudMapContainer[thisKeyInd] = globalCloud;
            }
        }

        // 4. Extract corner and edge from nearby key poses
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int)surroundingKeyPosesDS->size(); ++i)
        {
            if (pointDistance(surroundingKeyPosesDS->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadiusLarge)
                continue;
            
            int thisKeyInd = (int) surroundingKeyPosesDS->points[i].intensity;
            // already have been converted to global frame
            if (laserCloudFeatureMapContainer.find(thisKeyInd) != laserCloudFeatureMapContainer.end())
            {
                *laserCloudCornerFromMap += laserCloudFeatureMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudFeatureMapContainer[thisKeyInd].second;
            } else {
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]); 
                *laserCloudCornerFromMap += laserCloudCornerTemp; 
                *laserCloudSurfFromMap   += laserCloudSurfTemp; 
                laserCloudFeatureMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();

        if (laserCloudFeatureMapContainer.size() > 1000)
            laserCloudFeatureMapContainer.clear();

        ROS_WARN("ExtractNearby time: %fms, count: %d\n", tictoc.toc(), laserCloudFromMap->size());
    }

    void optimize()
    {
        TicToc tictoc;

        if (saveFrame() == false)
            return;

        if (timeFeatureIn.toSec() != timeLastIn)
            addOdomFactor();

        addPointFactor2();

        if (timeFeatureIn.toSec() == timeLastIn)
            return;

        // cout << "  iSAM2 Smoother Keys: " << endl;
        // for(const FixedLagSmoother2::KeyTimestampMap::value_type& key_timestamp: smootherISAM2->timestamps()) {
            // cout << setprecision(5) << "    Key: " << DefaultKeyFormatter(key_timestamp.first) << "  Time: " << key_timestamp.second << endl;
        // }

        IncrementalFixedLagSmoother2::Result result = smootherISAM2->update(factorGraph, initialEstimate, timestamps);
        factorGraph.print("Factor graphs\n");
        smootherISAM2->update();
        updateKeyframeMap(result);

        // printf("smoother update\n");
        // resize and reset 
        factorGraph.resize(0);
        initialEstimate.clear();
        timestamps.clear();

        // Values currentEstimate = isam->calculateEstimate();
        // currentEstimate.print("Final results:\n");
        // Pose3 latestEstimate = currentEstimate.at<Pose3>(Symbol('x', cloudKeyPoses6D->size()));
        Values currentEstimate = smootherISAM2->calculateEstimate();
        currentEstimate.print("Final results2: \n");
        Pose3 latestEstimate = currentEstimate.at<Pose3>(Symbol('x', cloudKeyPoses6D->size()));

        ROS_WARN("Optimization time: %fms\n", tictoc.toc());

        // save the latest
        latestX = latestEstimate.translation().x();
        latestY = latestEstimate.translation().y();
        latestZ = latestEstimate.translation().z();
        latestRoll = latestEstimate.rotation().roll();
        latestPitch = latestEstimate.rotation().pitch();
        latestYaw = latestEstimate.rotation().yaw();

        PointType thisPose3D;
        thisPose3D.x = latestX;
        thisPose3D.y = latestY;
        thisPose3D.z = latestZ;
        thisPose3D.intensity = cloudKeyPoses3D->size();
        cloudKeyPoses3D->push_back(thisPose3D);

        PointTypePose thisPose6D;
        thisPose6D.x = latestX;
        thisPose6D.y = latestY;
        thisPose6D.z = latestZ;
        thisPose6D.roll = latestRoll;
        thisPose6D.pitch = latestPitch;
        thisPose6D.yaw = latestYaw;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.time = timeFeatureIn.toSec();
        cloudKeyPoses6D->push_back(thisPose6D);
        updatePath(thisPose6D, globalPath);
        
        // temp
        PointTypePose tempPose;
        tempPose.x = featureInfo.initialGuessX;
        tempPose.y = featureInfo.initialGuessY;
        tempPose.z = featureInfo.initialGuessZ;
        tempPose.roll = featureInfo.initialGuessRoll;
        tempPose.pitch = featureInfo.initialGuessPitch;
        tempPose.yaw = featureInfo.initialGuessYaw;
        tempPose.time = timeFeatureIn.toSec();
        updatePath(tempPose, initialPath);

        ROS_WARN("Last optimized pose: %f %f %f %f %f %f", latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);


        if ()

        // save deskewed points
        pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudDeskewLast, *thisKeyFrame);
        deskewCloudKeyFrames.push_back(thisKeyFrame);

        // save feature points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        count++;
    }

    void updateKeyframeMap(IncrementalFixedLagSmoother2::Result result)
    {
        printf("Update keyframe map\n");
        for(auto const &data : result.marginalCovariances) 
        {
            string keyString = DefaultKeyFormatter(data.first);
            auto npos = keyString.find('l');
            if (npos != string::npos) // found landmark
            {
                int id = stoi(keyString.substr(npos+1));
                initialized[id] = false;
                marginalCovariances[id] = result.marginalCovariances[data.first];
                printf("Point %s marginalized\n", keyString.c_str());
            }
            else if (keyString.find('x') != string::npos) // found odometry pose
            {
                count--;
                printf("Odom %s marginalized\n", keyString.c_str());
            }
        }
    }

    bool saveFrame()
    {   
        if (cloudKeyPoses6D->points.empty())
            return true;

        if (featureId >= 0)
            return true;

        Eigen::Affine3f transStart = pclPoint2Affine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        ROS_WARN("Save frame -- roll: %f, pitch: %f, yaw: %f, dist: %f\n", roll, pitch, yaw, sqrt(x*x+y*y+z*z));
        return true;
    }

    gtsam::Pose3 float2gtsamPose(float x, float y, float z, float roll, float pitch, float yaw)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), 
                                  gtsam::Point3(x, y, z));
    }

    gtsam::Pose3 pclPoint2gtsamPose(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    Eigen::Affine3f pclPoint2Affine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }
};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_fusion");
    Optimization opt;
    

    ROS_INFO("\033[1;32m---->Optimization Started.\033[0m");

    std::thread optimizationThread(&Optimization::processOptimization, &opt);

    signal(SIGINT, signal_handle::signal_callback_handler);

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    optimizationThread.join();

    return 0;
}