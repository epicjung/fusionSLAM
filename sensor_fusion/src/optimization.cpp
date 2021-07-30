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

using namespace std;
using namespace gtsam;
using namespace cv;
using symbol_shorthand::C;
using symbol_shorthand::P;

typedef SmartProjectionPoseFactor<Cal3_S2> SmartFactor;

class optimization : public ParamServer
{
public:
    // GTSAM
    NonlinearFactorGraph factorGraph;
    Values initialEstimate;
    ISAM2 *isam;

    ros::Subscriber subPoint;
    ros::Publisher  pubPoint;
    ros::Publisher  pubTestImage;
    ros::Publisher  pubPath;
    ros::Publisher  pubOdomIncremental;
    int count;

    noiseModel::Diagonal::shared_ptr measNoise;
    noiseModel::Diagonal::shared_ptr poseNoise;
    noiseModel::Diagonal::shared_ptr pointNoise;
    noiseModel::Diagonal::shared_ptr odomNoise;

    map<int, int> numSeen;
    map<int, float> lastTime;
    map<int, bool> initialized;
    
    // float type
    float latestX;
    float latestY;
    float latestZ;
    float latestRoll;
    float latestPitch;
    float latestYaw;

    sensor_fusion::cloud_info cloudInfo;
    ros::Time cloudInfoTimeIn;

    Eigen::Affine3f initialAffine;
    Eigen::Affine3f lastAffine;

    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    nav_msgs::Path globalPath;


    optimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // Noises (TO-DO pose noise based on the params)
        // poseNoise = noiseModel::Isotropic::Sigma(9, 0.1); // 9 d.o.f
        measNoise   = noiseModel::Isotropic::Sigma(2, 1.0);// 1px in u and v
        poseNoise   = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        pointNoise  = noiseModel::Isotropic::Sigma(3, 2.0); // 3 d.o.f
        odomNoise   = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

        
        subPoint            = nh.subscribe<sensor_fusion::cloud_info>("lio_sam/feature/cloud_info", 1, &optimization::cloudHandler, this, ros::TransportHints().tcpNoDelay()); 
        pubPoint            = nh.advertise<visualization_msgs::MarkerArray>("fusion/feature/point_landmark", 1);
        pubTestImage        = nh.advertise<sensor_msgs::Image>("fusion/feature/test_image", 1);
        pubPath             = nh.advertise<nav_msgs::Path>("fusion/mapping/path", 1);
        pubOdomIncremental  = nh.advertise<nav_msgs::Odometry>("fusion/mapping/odometry_incremental", 1);
        
        // initialization
        count = 0;
        latestX = 0.0;
        latestY = 0.0;
        latestZ = 0.0;
        latestRoll = 0.0;
        latestPitch = 0.0;
        latestYaw = 0.0;
        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    }

    void updateInitialGuess()
    {
        lastAffine = pcl::getTransformation(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
        initialAffine = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);

        if (cloudKeyPoses6D->points.empty())
        {
            latestX = 0.0;
            latestY = 0.0;
            latestZ = 0.0;
            latestRoll = cloudInfo.imuRollInit;
            latestPitch = cloudInfo.imuPitchInit;
            latestYaw = cloudInfo.imuYawInit;
        } else {
            latestX = cloudInfo.initialGuessX;
            latestY = cloudInfo.initialGuessY;
            latestZ = cloudInfo.initialGuessZ;
            latestRoll = cloudInfo.initialGuessRoll;
            latestPitch = cloudInfo.initialGuessPitch;
            latestYaw = cloudInfo.initialGuessYaw;
        }
    }

    void addOdomFactor()
    {
        Pose3 poseTo = float2gtsamPose(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
        printf("addOdomFactor's key count: %d\n", cloudKeyPoses6D->points.size());
        if (cloudKeyPoses6D->points.empty())
        {
            factorGraph.add(PriorFactor<Pose3>(Symbol('x', 0), poseTo, poseNoise));
            initialEstimate.insert(Symbol('x', 0), poseTo);
        } else {
            Pose3 poseFrom = pclPoint2gtsamPose(cloudKeyPoses6D->points.back());
            int keyFrom = cloudKeyPoses6D->size() - 1;
            int keyTo = cloudKeyPoses6D->size();
            factorGraph.add(BetweenFactor<Pose3>(Symbol('x', keyFrom), Symbol('x', keyTo), 
                                                poseFrom.between(poseTo), odomNoise));
            initialEstimate.insert(Symbol('x', keyTo), poseTo);
        }
    }

    void addPointFactor()
    {
        printf("Is Cloud: %d\n", cloudInfo.isCloud);

        if (cloudInfo.isCloud)
            return;

        TicToc tictoc;

        int numPoint = 0;
        visualization_msgs::MarkerArray pointLandmark;
        cv::Mat testImg(ROW, COL, CV_8UC3, Scalar(255, 255, 255)); 
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

        cv_ptr->encoding = "bgr8";
        cv_ptr->header.stamp = cloudInfo.header.stamp;
        cv_ptr->header.frame_id = "image";
        cv_ptr->image = testImg;

        Pose3 camPose = float2gtsamPose(latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
  
        Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));

        for (size_t i = 0; i < cloudInfo.point_feature.points.size(); ++i)
        {
            Point2 measurement = Point2(double(cloudInfo.point_feature.points[i].x), 
                                        double(cloudInfo.point_feature.points[i].y));
            bool associated = !isinf(cloudInfo.point_feature.channels[1].values[i]);
            
            if (associated)
            {
                int id = int(cloudInfo.point_feature.channels[0].values[i]);
                // First point feature
                if (count == 0)
                {
                    // initialEstimate.insert(Symbol('l', id), point);
                    numSeen[id] = 1;
                }
                else
                {
                    if (numSeen.find(id) == numSeen.end()) // new feature
                    {
                        numSeen[id] = 1;
                    }
                    else // already exists
                    {
                        numSeen[id] += 1;

                        if (numSeen[id] >= 5)
                        {
                            printf("id %d Num seen: %d\n", id, numSeen[id]);
                            lastTime[id] = cloudInfoTimeIn.toSec();

                            // convert point to world frame
                            float localX = cloudInfo.point_feature.channels[1].values[i];
                            float localY = cloudInfo.point_feature.channels[2].values[i];
                            float localZ = cloudInfo.point_feature.channels[3].values[i];

                            // if (sqrt(localX*localX + localY*localY + localZ*localZ) < pointFeatureDistThres)
                            // {
                                float worldX = initialAffine(0,0)*localX+initialAffine(0,1)*localY+initialAffine(0,2)*localZ+initialAffine(0,3);
                                float worldY = initialAffine(1,0)*localX+initialAffine(1,1)*localY+initialAffine(1,2)*localZ+initialAffine(1,3);
                                float worldZ = initialAffine(2,0)*localX+initialAffine(2,1)*localY+initialAffine(2,2)*localZ+initialAffine(2,3);
                                Point3 worldPoint = Point3(worldX, worldY, worldZ);

                                // project point for checking
                                PinholeCamera<Cal3_S2> camera(camPose, *K);
                                Point2 estimation = camera.project(worldPoint);

                                // visualization (test image for projection)
                                cv::line(cv_ptr->image, cv::Point(measurement.x(), measurement.y()), cv::Point(estimation.x(), estimation.y()), Scalar(255, 0, 0), 1, LINE_8);
                            
                                cv::putText(cv_ptr->image, to_string(id), Point(measurement.x(), measurement.y()), 
                                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 255, 0), 0.5, cv::LINE_AA);

                                // visualization (off for operation)
                                visualization_msgs::Marker marker;
                                marker.action = 0;
                                marker.type = 2; // sphere
                                marker.ns = "points";
                                marker.id = id;
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

                                // add projection factor 
                                factorGraph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
                                    measurement, measNoise, Symbol('x', cloudKeyPoses6D->size()-1), Symbol('l', id), K);

                                if (!initialized[id])
                                {
                                    initialized[id] = true;
                                    initialEstimate.insert(Symbol('l', id), worldPoint);
                                    factorGraph.add(PriorFactor<Point3>(Symbol('l', id), worldPoint, pointNoise));
                                }
                                numPoint++;
                            // }
                        }
                    }
                }
            }
        }

        // Smart Factor

        printf("# of points: %d\n", numPoint);
        pubPoint.publish(pointLandmark);
        pubTestImage.publish(cv_ptr->toImageMsg());

        ROS_WARN("Add point factor time: %fms\n", tictoc.toc());
    }

    void updatePath(const PointTypePose& pose_in)
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
        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        nav_msgs::Odometry odomIncre;
        odomIncre.header.stamp = cloudInfoTimeIn;
        odomIncre.header.frame_id = odometryFrame;
        odomIncre.child_frame_id = "odom_mapping";
        odomIncre.pose.pose.position.x = latestX;
        odomIncre.pose.pose.position.y = latestY;
        odomIncre.pose.pose.position.z = latestZ;
        odomIncre.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(latestRoll, latestPitch, latestYaw);
        pubOdomIncremental.publish(odomIncre);
        // TO-DO: consider loop closing and publish relative pose
    }  

    void publishFrames()
    {
        if (cloudKeyPoses6D->points.empty())
            return;

        // publish key poses
        static tf::TransformBroadcaster tfMap2Cam;
        tf::Transform map2Cam = tf::Transform(tf::createQuaternionFromRPY(latestRoll, latestPitch, latestYaw), 
                                              tf::Vector3(latestX, latestY, latestZ));
        tfMap2Cam.sendTransform(tf::StampedTransform(map2Cam, cloudInfoTimeIn, mapFrame, camFrame));

        // publish global path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = cloudInfoTimeIn;
            globalPath.header.frame_id = mapFrame;
            pubPath.publish(globalPath);
        }
    }

    void cloudHandler(const sensor_fusion::cloud_info cloudInfoMsg)
    {
        ROS_WARN("Cloud Handler");
        
        cloudInfo = cloudInfoMsg;
        cloudInfoTimeIn = cloudInfo.header.stamp;

        Cal3_S2::shared_ptr K(new Cal3_S2(230.39028930664062, 230.31454467773438, 0.0, 239.93666076660156, 136.52784729003906));

        // Fix the first id?
        printf("Count: %d\n", count);

        updateInitialGuess();

        if (saveFrame() == false)
            return;

        addOdomFactor();
        
        addPointFactor();

        optimize();

        publishOdometry();

        publishFrames();

        count++;
    }

    void optimize()
    {
        TicToc tictoc;

        // optimize and get results
        initialEstimate.print("Initial results:\n");
        isam->update(factorGraph, initialEstimate);
        isam->update();

        // resize and reset 
        factorGraph.resize(0);
        initialEstimate.clear();

        Values currentEstimate = isam->calculateEstimate();
        currentEstimate.print("Final results:\n");
        Pose3 latestEstimate = currentEstimate.at<Pose3>(Symbol('x', cloudKeyPoses6D->size()));
        
        ROS_WARN("Optimization time: %fms\n", tictoc.toc());

        // save the latest
        latestX = latestEstimate.translation().x();
        latestY = latestEstimate.translation().y();
        latestZ = latestEstimate.translation().z();
        latestRoll = latestEstimate.rotation().roll();
        latestPitch = latestEstimate.rotation().pitch();
        latestYaw = latestEstimate.rotation().yaw();

        PointTypePose thisPose6D;
        thisPose6D.x = latestX;
        thisPose6D.y = latestY;
        thisPose6D.z = latestZ;
        thisPose6D.roll = latestRoll;
        thisPose6D.pitch = latestPitch;
        thisPose6D.yaw = latestYaw;
        thisPose6D.time = cloudInfoTimeIn.toSec();
        cloudKeyPoses6D->push_back(thisPose6D);
        updatePath(thisPose6D);
    }

    bool saveFrame()
    {
        if (cloudKeyPoses6D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPoint2Affine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = initialAffine;
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        printf("Save frame -- roll: %f, pitch: %f, yaw: %f, dist: %f\n", roll, pitch, yaw, sqrt(x*x+y*y+z*z));
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
    optimization opt;

    ROS_INFO("\033[1;32m---->Optimization Started.\033[0m");

    // std::thread optimizationThread(&optimization::initCallback, &opt);

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    // optimizationThread.join();

    return 0;
}