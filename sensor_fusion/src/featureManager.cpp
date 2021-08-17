#include "featureManager.h"

const int queueLength = 2000;

int FeaturePerId::endFrame()
{
    return startFrame + featurePerFrame.size() - 1;
}

FeatureManager::FeatureManager()
{
    pubImgProjected = nh.advertise<sensor_msgs::Image>("/fusion/visual/projected_img", 1);

    allocateMemory();
}  

void FeatureManager::allocateMemory()
{
    currentCloud.reset(new pcl::PointCloud<PointType>());
    laserCloudNearby.reset(new pcl::PointCloud<PointType>());

    //     // get camera info
    // for (size_t i = 0; i < CAM_NAMES.size(); i++)
    // {
    //     ROS_DEBUG("reading paramerter of camera %s", CAM_NAMES[i].c_str());
    //     FILE *fh = fopen(CAM_NAMES[i].c_str(), "r");
    //     if (fh == NULL)
    //     {
    //         ROS_WARN("config_file doesn't exist");
    //         ROS_BREAK();
    //         return;
    //     }
    //     fclose(fh);

    //     camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[i]);
    //     m_camera.push_back(camera);
    // }

    resetParameters();
}

void FeatureManager::resetParameters()
{
    currentCloud->clear();
    timeSent = 0.0;
    frameCount = 0;
    keyframeCount = 0;
}

void FeatureManager::inputCloudNearby(const pcl::PointCloud<PointType>::Ptr cloudIn)
{
    lock_guard<mutex> lock(lidarLock);
    laserCloudNearby->clear();
    pcl::copyPointCloud(*cloudIn, *laserCloudNearby);
    printf("CloudNearby size: %d\n", (int)laserCloudNearby->size());
}

void FeatureManager::inputIMU(const sensor_msgs::Imu imuIn)
{
    std::lock_guard<std::mutex> lock(imuLock);
    imuQueue.push_back(imuIn);
}

void FeatureManager::inputPointFeature(const sensor_msgs::PointCloud pointIn)
{
    TicToc tictoc;
    pointFeature = pointIn;
    timeImageCur = ROS_TIME(&pointIn);

    if (!updateInitialPose())
    {
        ROS_WARN("PointHandler: updateInitialPose failure");
        return;
    }

    if (addPointFeature())
    {
        printf("addPointFeautre: imu - %d, odom - %d\n", imuAvailable, odomAvailable);

        associatePointFeature();

        addKeyframe();
        
        frameCount++;
    } 
    else 
    {
        printf("addPointFeautre: fail\n");
        frameCount++;
        return;
    }
    ROS_WARN("FM: frameCount: %d\n", frameCount);
    ROS_WARN("Manage point time: %fms\n", tictoc.toc());
}

void FeatureManager::addKeyframe()
{
    keyframeCount = frameCount;

    for (auto &it : pointFeaturesPerId)
    {
        if (keyframeCount - it.startFrame < (int)it.featurePerFrame.size())
        {
            printf("Key %d: StartFrame: %d, keyFrame: %d, frame: %d, fpf size: %d\n", it.featureId, it.startFrame, keyframeCount, frameCount, (int)it.featurePerFrame.size());
            sensor_fusion::cloud_info pointInfo = it.featurePerFrame.at(keyframeCount - it.startFrame);
            featLock.lock();
            featureQueue.push_back(make_pair(pointInfo.featureId, pointInfo));
            featLock.unlock();
        }
    }
}
    
void FeatureManager::inputCloudInfo(const sensor_fusion::cloud_info cloudInfoIn)
{
    cloudInfo = cloudInfoIn;

    timeCloudInfoCur = ROS_TIME(&cloudInfoIn);
    
    manageCloudFeature();

    // TO-DO: Feature extraction and publish
    featLock.lock();
    featureQueue.push_back(make_pair(-1, cloudInfo)); // id = -1 for cloudInfo
    featLock.unlock();
}

void FeatureManager::inputOdom(const nav_msgs::Odometry odomIn)
{
    lock_guard<mutex> lock(odomLock);
    odomQueue.push_back(odomIn);
}

void FeatureManager::inputImage(const sensor_msgs::Image imgIn)
{
    lock_guard<mutex> lock(imgLock);
    imgQueue.push_back(make_pair(ROS_TIME(&imgIn), imgIn));
}

void FeatureManager::removeOldFeatures(double refTime)
{
    for (auto it = pointFeaturesPerId.begin(), it_next = pointFeaturesPerId.begin(); it != pointFeaturesPerId.end(); it = it_next)
    {
        it_next++;

        int lastIdx = it->endFrame() - it->startFrame;

        // printf("id: %d, startFrame: %d, size: %d, lastIdx: %d\n", it->featureId, it->startFrame, (int)it->featurePerFrame.size(), lastIdx);

        if (it->featurePerFrame.at(lastIdx).header.stamp.toSec() < refTime)
        {
            pointLock.lock();
            pointFeaturesPerId.erase(it);
            pointLock.unlock();
        }
    }

    while(!featureQueue.empty())
    {
        sensor_fusion::cloud_info pointInfo = featureQueue.front().second;
        if (pointInfo.header.stamp.toSec() < refTime)
        {
            featLock.lock();
            featureQueue.pop_front();
            featLock.unlock();
        }
        else
            break;
    }
}

bool FeatureManager::addPointFeature() // check keyframe and add if it is a keyframe
{
    TicToc tictoc;
    float parallaxSum = 0;
    int parallaxNum = 0;
    int lastTrackNum = 0;
    int newFeatureNum = 0;
    int longTrackNum = 0;
    bool isKey = false;

    for (int i = 0; i < (int)pointFeature.points.size(); ++i)
    {
        int featureId = (int) pointFeature.channels[0].values[i];

        sensor_fusion::cloud_info pointInfo;
        pointInfo.header.stamp = ros::Time().fromSec(timeImageCur);
        pointInfo.header.frame_id = camFrame;
        pointInfo.initialGuessX = latestX;
        pointInfo.initialGuessY = latestY;
        pointInfo.initialGuessZ = latestZ;
        pointInfo.initialGuessRoll = latestRoll;
        pointInfo.initialGuessPitch = latestPitch;
        pointInfo.initialGuessYaw = latestYaw;
        pointInfo.imuRollInit = latestImuRoll;
        pointInfo.imuPitchInit = latestImuPitch;
        pointInfo.imuYawInit = latestImuYaw;
        pointInfo.odomAvailable = odomAvailable;
        pointInfo.imuAvailable = imuAvailable;
        pointInfo.point2d.x = pointFeature.points[i].x;
        pointInfo.point2d.y = pointFeature.points[i].y;
        pointInfo.point2d.z = pointFeature.points[i].z;
        pointInfo.uv.x = pointFeature.channels[4].values[i];
        pointInfo.uv.y = pointFeature.channels[5].values[i];
        pointInfo.featureId = featureId;
        pointInfo.estimatedDepth = -1.0;

        auto it = find_if(pointFeaturesPerId.begin(), pointFeaturesPerId.end(), [featureId](const FeaturePerId &it)
                    {
                        return it.featureId == featureId;
                    });
        if (it == pointFeaturesPerId.end())
        {
            pointFeaturesPerId.push_back(FeaturePerId(featureId, frameCount, timeImageCur));
            pointFeaturesPerId.back().featurePerFrame.push_back(pointInfo);
            newFeatureNum++;
        }
        else if (it->featureId == featureId)
        {
            it->featurePerFrame.push_back(pointInfo);
            lastTrackNum++;
            if (it->featurePerFrame.size() >= 4)
                longTrackNum++;
        }
    }

    // lastTrackNum: # of features that have already existed previously --> smaller means there are more fresh features = keyframe
    // longTrackNum: # of features that have existed for more than 4 frames --> smaller means there are more fresh features = keyframe
    // newFeatureNum > 0.5 * lastTrackNum: more fresh feature are there than 0.5 * those have been tracked before  
    printf("key: %d, frame: %d, lastTrackNum: %d, longTrackNum: %d, newFeatureNum: %d\n", keyframeCount, frameCount, lastTrackNum, longTrackNum, newFeatureNum);
    if (frameCount < 2 || lastTrackNum < 20 || longTrackNum < 40 || newFeatureNum > 0.5 * lastTrackNum)  
    {
        isKey = true;
    }
    else
    {
        for (auto &it : pointFeaturesPerId)
        {    
            if (it.startFrame <= keyframeCount - 2 && it.startFrame + int(it.featurePerFrame.size()) > frameCount)
            {
                parallaxSum += compensatedParallax2(it);
                parallaxNum++;
            }
        }

        if (parallaxNum == 0)
        {
            isKey = true;
        }
        else {
            // printf("parallax_sum: %lf, parallax_num: %d\n", parallaxSum, parallaxNum);
            // printf("current parallax: %lf\n", parallaxSum / parallaxNum);
            isKey = parallaxSum / parallaxNum >= MIN_PARALLAX;
        }
    }

    ROS_WARN("Add point feature time: %fms\n", tictoc.toc());

    return isKey;
}

float FeatureManager::compensatedParallax2(const FeaturePerId &it)
{
    // printf("%d: StartFrame: %d, keyFrame: %d, frame: %d, fpf size: %d\n", it.featureId, it.startFrame, keyframeCount, frameCount, (int)it.featurePerFrame.size());

    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const sensor_fusion::cloud_info &frame_i = it.featurePerFrame[keyframeCount - 1 - it.startFrame];
    const sensor_fusion::cloud_info &frame_j = it.featurePerFrame[frameCount - it.startFrame];

    float ans = 0;
    Vector3f p_j(frame_j.point2d.x, frame_j.point2d.y, frame_j.point2d.z);

    float u_j = p_j(0);
    float v_j = p_j(1);

    Vector3f p_i(frame_i.point2d.x, frame_i.point2d.y, frame_i.point2d.z);
    Vector3f p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    float dep_i = p_i(2);
    float u_i = p_i(0) / dep_i;
    float v_i = p_i(1) / dep_i;
    float du = u_i - u_j, dv = v_i - v_j;

    float dep_i_comp = p_i_comp(2);
    float u_i_comp = p_i_comp(0) / dep_i_comp;
    float v_i_comp = p_i_comp(1) / dep_i_comp;
    float du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}


void FeatureManager::updateImuRPY()
{
    // use imu orientation
    imuAvailable = false;

    while (!imuQueue.empty())
    {
        if (ROS_TIME(&imuQueue.front()) < timeImageCur - 0.01)
            imuQueue.pop_front();
        else
            break;
    }

    if (imuQueue.empty())
        return;

    for (int i = 0; i < (int) imuQueue.size(); ++i)
    {
        sensor_msgs::Imu thisImu = imuQueue[i];
        double timeImuCur = ROS_TIME(&thisImu);
        if (timeImuCur <= timeImageCur)
            imuRPY2rosRPY(&thisImu, &latestImuRoll, &latestImuPitch, &latestImuYaw);
        
        if (timeImuCur > timeImageCur)
        {
            imuAvailable = true;
            break;
        }
    }

    if (imuAvailable)
    {
        Eigen::Affine3f imuInitRPY = pcl::getTransformation(0.0, 0.0, 0.0, latestImuRoll, latestImuPitch, latestImuYaw);
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(affineL2C, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f transformIncre = pcl::getTransformation(0.0, 0.0, 0.0, roll, pitch, yaw);
        Eigen::Affine3f camInitRPY = imuInitRPY * transformIncre;
        pcl::getTranslationAndEulerAngles(camInitRPY, x, y, z, latestImuRoll, latestImuPitch,  latestImuYaw);
    }
}

void FeatureManager::updateOdometry()
{
    odomAvailable = false;

        // get odometry 
    while (!odomQueue.empty())
    {
        if (ROS_TIME(&odomQueue.front()) < timeImageCur - 0.01)
            odomQueue.pop_front();
        else
            break;
    }
    if (odomQueue.empty())
        return;

    if (ROS_TIME(&odomQueue.front()) > timeImageCur)
        return;

    nav_msgs::Odometry startOdomMsg;
    
    for(int i = 0; i < (int)odomQueue.size(); ++i)
    {
        startOdomMsg = odomQueue[i];
        if (ROS_TIME(&startOdomMsg) < timeImageCur)
        {
            continue;
        } else {
            odomAvailable = true;
            break;
        }
    }

    if (odomAvailable)
    {
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
        
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        ROS_WARN("FM time: %f\n%f, %f, %f, %f, %f, %f", timeImageCur, startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,  startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        transWorld2Cam = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z,
                                            (float) roll, (float) pitch, (float) yaw) * affineL2C;
        pcl::getTranslationAndEulerAngles(transWorld2Cam, latestX, latestY, latestZ, latestRoll, latestPitch, latestYaw);
    }
}

bool FeatureManager::updateInitialPose()
{
    lock_guard<mutex> lock1(imuLock);
    lock_guard<mutex> lock2(odomLock);

    if (imuQueue.empty() || ROS_TIME(&imuQueue.front()) > timeImageCur)
    {
        ROS_DEBUG("Waiting for IMU data ...");
        return false;
    }

    updateImuRPY();

    updateOdometry();
    
    return true;
}

void FeatureManager::manageCloudFeature()
{
    TicToc tictoc;
    if (cloudInfo.imuAvailable)
    {        
        Eigen::Affine3f imuInitRPY = pcl::getTransformation(0.0, 0.0, 0.0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(affineL2C, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f transformIncre = pcl::getTransformation(0.0, 0.0, 0.0, roll, pitch, yaw);
        Eigen::Affine3f camInitRPY = imuInitRPY * transformIncre;
        pcl::getTranslationAndEulerAngles(camInitRPY, x, y, z, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
    }
    
    if (cloudInfo.odomAvailable)
    {
        Eigen::Affine3f transInit = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ, 
                                                cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw) * affineL2C;
        pcl::getTranslationAndEulerAngles(transInit, cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                        cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
    }

    // transform deskewed points
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
    PointTypePose transformPose = trans2PointTypePose(affineL2C.inverse());
    *cloudOut = *transformPointCloud(cloudOut, &transformPose);
    pcl::toROSMsg(*cloudOut, cloudInfo.cloud_deskewed);
    publishCloud(&pubExtractedCloudCam, cloudOut, cloudInfo.header.stamp, camFrame);
    ROS_WARN("Manage cloud feature time: %fms\n", tictoc.toc());
}

void FeatureManager::associatePointFeature()
{
    if (!odomAvailable)
        return;
    TicToc tictoc;

    lock_guard<mutex> lock(lidarLock);

    // sensor_msgs::ChannelFloat32 depths;
    // depths.name = "depth";
    // depths.values.resize(pointFeature.points.size(), -1); // -1 for initial depth
    int count = 0;

    if (laserCloudNearby->size() == 0)
    {
        printf("No cloud nearby\n");
        return;
    }

    printf("laserCloudNearby: %d\n", laserCloudNearby->size());

    // 1. Transform pointcloud to camera frame
    pcl::PointCloud<PointType>::Ptr localCloud(new pcl::PointCloud<PointType>());
    PointTypePose transformPose = trans2PointTypePose(transWorld2Cam.inverse());
    *localCloud = *transformPointCloud(laserCloudNearby, &transformPose);

    // 2. Project depth cloud on the range image and filter points in the same region
    int numBins = 360; 
    vector<vector<PointType>> pointArray;
    pointArray.resize(numBins);
    for (int i = 0; i < numBins; ++i)
        pointArray[i].resize(numBins);
    float angRes = 180.0 / float(numBins); // cover only -90~90 FOV of the lidar 
    cv::Mat rangeImage = cv::Mat(numBins, numBins, CV_32F, cv::Scalar::all(FLT_MAX));

    printf("2. Project depth cloud on the range image and filter points in the same region\n");
    int temp = 0;

    for (int i = 0; i < (int)localCloud->size(); ++i)
    {
        PointType p = localCloud->points[i];
        if (p.z > 0.0 && abs(p.y / p.z) < 10.0 && abs(p.x / p.z) < 10.0)
        {
            temp++;
            // Horizontal: 0 [deg] (x > 0) ~ 180 [deg] (x < 0) w.r.t x-axis
            float horizonAngle = atan2(p.z, p.x) * 180.0 / M_PI; 
            int colIdx = round(horizonAngle / angRes);

            // Vertical: -90 [deg] (y < 0) ~ 90 [deg] (y > 0) w.r.t xz plane
            float verticalAngle = atan2(p.y, sqrt(p.x*p.x+p.z*p.z)) * 180.0 / M_PI;
            int rowIdx = round((verticalAngle + 90.0) / angRes);

            if (colIdx >= 0 && colIdx < numBins && rowIdx >= 0 && rowIdx < numBins)
            {
                float range = pointDistance(p);
                if (range < rangeImage.at<float>(rowIdx, colIdx))
                {
                    rangeImage.at<float>(rowIdx, colIdx) = range; 
                    pointArray[rowIdx][colIdx] = p; 
                }
            }
        }
    }
    printf("Included %d\n", temp);

    printf("3. localCloud size: %d\n", (int)localCloud->size());


    // 3. Extract cloud from the range image
    pcl::PointCloud<PointType>::Ptr localCloudFiltered(new pcl::PointCloud<PointType>());
    for (int i = 0; i < numBins; ++i)
    {
        for (int j = 0; j < numBins; ++j)
        {
            if (rangeImage.at<float>(i,j) != FLT_MAX)
                localCloudFiltered->push_back(pointArray[i][j]);
        }
    }

    printf("3. localCloudFiltered size: %d\n", (int)localCloudFiltered->size());

    // 4. Project pointcloud onto a unit sphere
    pcl::PointCloud<PointType>::Ptr localCloudSphere(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)localCloudFiltered->size(); ++i)
    {
        PointType p = localCloudFiltered->points[i];
        float range = pointDistance(p);
        p.x /= range;
        p.y /= range;
        p.z /= range;
        p.intensity = range;
        localCloudSphere->push_back(p);
    }
    printf("4. Project pointcloud onto a unit sphere\n");

    // TO-DO: cloudInfo LiDAR로 변경 필요
    if (localCloudSphere->size() < 10) 
        return;

    printf("localCloudSphere->size() < 10\n");

    // No need
    cv_bridge::CvImagePtr cv_ptr;

    while (!imgQueue.empty())
    {
        if (imgQueue.front().first < timeImageCur)
        {
            imgQueue.pop_front();
            printf("Old images\n");
        }
        else if (imgQueue.front().first == timeImageCur)
        {
            // cv::Mat grayImage = imgQueue.front().second;
            // cvtColor(grayImage, thisImage, CV_GRAY2RGB);
            sensor_msgs::Image thisImage = imgQueue.front().second;
            cv_ptr = cv_bridge::toCvCopy(thisImage, sensor_msgs::image_encodings::BGRA8);
            imgQueue.pop_front();
            printf("Found the image\n");
            break;
        }
    }
    
    printf("Noneed \n");

    // 5. Project 2D point feature to unit sphere
    pcl::PointCloud<PointType>::Ptr pointSphere(new pcl::PointCloud<PointType>());
    std::vector<FeaturePerId*> iterCandidate; 
    printf("point feature counter: %d\n", (int)pointFeaturesPerId.size());      
    for (auto &it : pointFeaturesPerId)
    {            
        int lastIndex = it.startFrame + it.featurePerFrame.size() - 1;
        printf("id: %d, lastIndex: %d, key: %d\n", it.featureId, lastIndex, keyframeCount);

        if ((int) it.featurePerFrame.size() >= 2 && lastIndex == keyframeCount)
        {            
            Eigen::Vector3f thisFeature(it.featurePerFrame.back().point2d.x, 
                                        it.featurePerFrame.back().point2d.y, 
                                        it.featurePerFrame.back().point2d.z);
            thisFeature.normalize();
            PointType p;
            p.x = thisFeature(0);
            p.y = thisFeature(1);
            p.z = thisFeature(2);
            p.intensity = -1; // depth
            iterCandidate.push_back(&it);
            pointSphere->push_back(p);
            if (!it.isDepth)
            {
                cv::Point2f uv(it.featurePerFrame.back().uv.x, it.featurePerFrame.back().uv.y);
                cv::circle(cv_ptr->image, uv, 3, cv::Scalar(0, 0, 255), -1);
            }
            else
            {
                cv::Point2f uv(it.featurePerFrame.back().uv.x, it.featurePerFrame.back().uv.y);
                cv::circle(cv_ptr->image, uv, 3, cv::Scalar(0, 255, 255), -1);
            }
        }
    }

    // for (int i = 0; i < (int)pointFeature.points.size(); ++i)
    // {
    //     Eigen::Vector3f thisFeature(pointFeature.points[i].x, pointFeature.points[i].y, 1.0);
    //     thisFeature.normalize();
    //     PointType p;
    //     p.x = thisFeature(0);
    //     p.y = thisFeature(1);
    //     p.z = thisFeature(2);
    //     p.intensity = -1; // depth
    //     pointSphere->push_back(p);
    // }

    // 6. create kd-tree
    pcl::KdTreeFLANN<PointType>::Ptr kdTree(new pcl::KdTreeFLANN<PointType>());
    kdTree->setInputCloud(localCloudSphere);

    pcl::PointCloud<PointType>::Ptr threePoints(new pcl::PointCloud<PointType>());
    
    // 7. find feature depth using kd-tree
    vector<int> pointSearchIdx;
    vector<float> pointSearchSquaredDist;
    float distThres = pow(sin(angRes / 180.0 * M_PI) * depthAssociateBinNumber, 2); 
    for (int i = 0; i < (int)pointSphere->size(); ++i)
    {
        PointType p = pointSphere->points[i];
        kdTree->nearestKSearch(p, 3, pointSearchIdx, pointSearchSquaredDist);
        // Three nearest are found and the farthest is within the threshold
        // DEBUG
        threePoints->push_back(localCloudSphere->points[pointSearchIdx[0]]);      
        threePoints->push_back(localCloudSphere->points[pointSearchIdx[1]]);      
        threePoints->push_back(localCloudSphere->points[pointSearchIdx[2]]);      

        if (pointSearchIdx.size() == 3 && pointSearchSquaredDist[2] < distThres)
        {
            float x0 = localCloudSphere->points[pointSearchIdx[0]].x;
            float y0 = localCloudSphere->points[pointSearchIdx[0]].y;
            float z0 = localCloudSphere->points[pointSearchIdx[0]].z;
            float r0 = localCloudSphere->points[pointSearchIdx[0]].intensity;
            Vector3f A(x0*r0, y0*r0, z0*r0);

            float x1 = localCloudSphere->points[pointSearchIdx[1]].x;
            float y1 = localCloudSphere->points[pointSearchIdx[1]].y;
            float z1 = localCloudSphere->points[pointSearchIdx[1]].z;
            float r1 = localCloudSphere->points[pointSearchIdx[1]].intensity;
            Vector3f B(x1*r1, y1*r1, z1*r1);

            float x2 = localCloudSphere->points[pointSearchIdx[2]].x;
            float y2 = localCloudSphere->points[pointSearchIdx[2]].y;
            float z2 = localCloudSphere->points[pointSearchIdx[2]].z;
            float r2 = localCloudSphere->points[pointSearchIdx[2]].intensity;
            Vector3f C(x2*r2, y2*r2, z2*r2);

            Vector3f V(p.x, p.y, p.z);
            Vector3f N = (A-B).cross(B-C);
            
            float depth = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) 
                            / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));
            float minDepth = min(r0, min(r1, r2));
            float maxDepth = max(r0, max(r1, r2));
            
            if (maxDepth - minDepth > 2 || depth <= 0.5)
                continue;
            else if (depth > maxDepth)
                depth = maxDepth;
            else if (depth < minDepth)
                depth = minDepth;

            // de-normalized the 3D unit sphere feature
            pointSphere->points[i].x *= depth;
            pointSphere->points[i].y *= depth;
            pointSphere->points[i].z *= depth;
            pointSphere->points[i].intensity = pointSphere->points[i].z;
        
            // update 3D point feature (sensor_msgs::PointCloud)
            if (pointSphere->points[i].intensity > lidarMinRange && pointSphere->points[i].intensity < lidarMaxRange)
            {
                iterCandidate[i]->featurePerFrame.back().estimatedDepth = pointSphere->points[i].intensity;
                iterCandidate[i]->featurePerFrame.back().point3d.x = pointSphere->points[i].x;
                iterCandidate[i]->featurePerFrame.back().point3d.y = pointSphere->points[i].y;
                iterCandidate[i]->featurePerFrame.back().point3d.z = pointSphere->points[i].z;
                printf("id: %d, cur: %d, keyframe: %d, depth: %f\n", iterCandidate[i]->featureId, iterCandidate[i]->featurePerFrame.size()-1, keyframeCount, iterCandidate[i]->featurePerFrame.back().estimatedDepth);
                // depths.values[i] = pointSphere->points[i].intensity;
                if (iterCandidate[i]->isDepth == false)
                    iterCandidate[i]->isDepth = true;
                // ROS_WARN("Association Status");
                // printf("%d: xyz -> %f, %f, %f; uv -> %f, %f\n", 
                //         iterCandidate[i]->featureId, 
                //         iterCandidate[i]->estimatedXYZ.x(),
                //         iterCandidate[i]->estimatedXYZ.y(),
                //         iterCandidate[i]->estimatedXYZ.z(),
                //         iterCandidate[i]->featurePerFrame.back().uv.x(), 
                //         iterCandidate[i]->featurePerFrame.back().uv.y());

                cv::Point2f uv(iterCandidate[i]->featurePerFrame.back().uv.x, iterCandidate[i]->featurePerFrame.back().uv.y);
                cv::circle(cv_ptr->image, uv, 3, cv::Scalar(0, 255, 0), -1);
            
                count++;
                // pointFeature.channels[1].values[i] = pointSphere->points[i].x;
                // pointFeature.channels[2].values[i] = pointSphere->points[i].y;
                // pointFeature.channels[3].values[i] = pointSphere->points[i].z;
            }
        }
    }
    pubImgProjected.publish(cv_ptr->toImageMsg());

    // visualizeAssociatedPoints(depths, localCloud);

    /* DEBUG */
    ROS_WARN("Associate point feature time: %fms\n", tictoc.toc());
    printf("PointFeature size: %d\n", (int)iterCandidate.size());
    printf("AssociatedFeature size: %d\n", count);
}

void FeatureManager::visualizeAssociatedPoints(const sensor_msgs::ChannelFloat32 depths, const pcl::PointCloud<PointType>::Ptr localCloud)
{
    // pop old images
    cv_bridge::CvImagePtr cv_ptr;
    bool isFound = false;
    {
        lock_guard<mutex> lock4(imgLock);
        
        while (!imgQueue.empty())
        {
            if (imgQueue.front().first < timeImageCur)
            {
                imgQueue.pop_front();
            }
            else if (imgQueue.front().first == timeImageCur)
            {
                // cv::Mat grayImage = imgQueue.front().second;
                // cvtColor(grayImage, thisImage, CV_GRAY2RGB);
                sensor_msgs::Image thisImage = imgQueue.front().second;
                cv_ptr = cv_bridge::toCvCopy(thisImage, sensor_msgs::image_encodings::BGRA8);
                imgQueue.pop_front();
                isFound = true;
                printf("Found the image\n");
                break;
            }
        }
    }

    if (isFound)
    {
        // // project 3d to image plane
        // for (auto& point : localCloud->points)
        // {
        //     if (point.z > 0)
        //     {
        //         Vector3d spacePoint(point.x, point.y, point.z);
        //         float maxVal = 20.0;
        //         int red = min(255, (int)(255 * abs((point.z - maxVal) / maxVal)));
        //         int green = min(255, (int)(255 * (1 - abs((point.z - maxVal) / maxVal ))));
        //         Vector2d imagePoint;
        //         m_camera[0]->spaceToPlane(spacePoint, imagePoint);
        //         if (imagePoint.x() >= 0 && imagePoint.x() < m_camera[0]->imageWidth() && imagePoint.y() >=0 && imagePoint.y() < m_camera[0]->imageHeight())
        //         {
        //             // printf("3d: %f, %f, %f\n", point.x, point.y, point.z);
        //             // printf("uv: %f, %f\n", imagePoint.x(), imagePoint.y());
        //             // printf("red: %d, green %d\n", red, green);
        //             cv::Point2f uv(imagePoint.x(), imagePoint.y());
        //             cv::circle(cv_ptr->image, uv, 0.5, cv::Scalar(0, green, red), -1);
        //         }
        //     }
        // }

        for (int i = 0; i < pointFeature.points.size(); ++i)
        {
            cv::Point2f uv(pointFeature.channels[4].values[i], pointFeature.channels[5].values[i]);
            if (depths.values[i] >= 0)
            {
                cv::circle(cv_ptr->image, uv, 3, cv::Scalar(0, 255, 0), -1);
            }
            else
            {
                cv::circle(cv_ptr->image, uv, 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        pubImgProjected.publish(cv_ptr->toImageMsg());
    }
}



// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "sensor_fusion");

//     FeatureManager FE;

//     std::thread processThread(&FeatureManager::processFeature, &FE);

//     signal(SIGINT, signal_handle::signal_callback_handler);

//     ROS_INFO("\033[1;32m----> Feature Manager Started.\033[0m");

//     ros::MultiThreadedSpinner spinner(3);
//     spinner.spin();

//     processThread.join();

//     return 0;
// }