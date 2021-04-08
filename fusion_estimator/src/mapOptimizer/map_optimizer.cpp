
#include "map_optimizer.h"

MapOptimizer::MapOptimizer()
{
	isDegenerate = false;

	laserCloudCornerFromMapDSNum = 0;
	laserCloudSurfFromMapDSNum = 0;
	laserCloudCornerLastDSNum = 0;
	laserCloudSurfLastDSNum = 0;

	aLoopIsClosed = false;
}

MapOptimizer::~MapOptimizer() {}

void MapOptimizer::allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
    
    for (int i = 0; i < 6; ++i){
        transformTobeMapped[i] = 0;
    }
}

void MapOptimizer::setCloudInfo(const fusion_estimator::CloudInfo &cloudInfoIn)
{
	cloudInfo = std::move(cloudInfoIn);

	rosTimeCloudInfo = cloudInfo.header.stamp;
	timeCloudInfo = rosTimeCloudInfo.toSec();

	pcl::fromROSMsg(cloudInfo.cloud_corner, *laserCloudCornerLast);
	pcl::fromROSMsg(cloudInfo.cloud_surface, *laserCloudSurfLast);
}

void MapOptimizer::updateInitialGuess()
{
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastImuTransformation;
    // initialization
    if (cloudKeyPoses3D->points.empty())
    {
        transformTobeMapped[0] = cloudInfo.imuRollInit;
        transformTobeMapped[1] = cloudInfo.imuPitchInit;
        transformTobeMapped[2] = cloudInfo.imuYawInit;

        if (!useImuHeadingInitialization)
            transformTobeMapped[2] = 0;

        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
        return;
    }	
}

void MapOptimizer::extractSurroundingKeyFrames()
{
    if (cloudKeyPoses3D->points.empty() == true)
        return; 
   
    // extractNearby();
}

void MapOptimizer::downsampleCurrentScan()
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

void scan2MapOptimization()
{
    if (cloudKeyPoses3D->points.empty())
        return;
}

void saveKeyFramesAndFactor()
{
    if (saveFrame() == false)
        return;
    //saveFrame() = true if cloudKeyPoses3D->points is empty

    // odom factor (graph-optimized된 k 위치와 현재 map-optimized된 k+1 위치의 차이를 betweenFactor로 놂)
    addOdomFactor();

    // gps factor (just factor)
    addGPSFactor();

    // loop factor (betweenFactor)
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }

    // resize and reset
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1); // get last value
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D); //cloudKeyPoses3D has xyz only

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D); //cloudKeyPoses6D has xyz and rotations

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

    // save key frame cloud (각 keyframe에서의 corner, surface points 갖고 있음)
    // loop-closure thread가 돌아가면서 icp를 할 때 필요함
    // thisPose6D, thisPose3D도 결국에는 loop-closure를 할 때 필요함 
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
}