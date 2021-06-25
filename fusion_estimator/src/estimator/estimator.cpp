#include "estimator.h"

Estimator::Estimator()
{
	initThreadFlag= false;
    initCloudFlag = false;
    imageDeskewFlag = false;
	deskewFlag = 0;
	cloudFrameCount = 0;
    prevImgTime = -1.0;
    key = 1;

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    optimizer = new ISAM2(parameters);

    laserCloudCornerFromMapDSNum = 0;
    laserCloudSurfFromMapDSNum = 0;
    laserCloudCornerDSNum = 0;
    laserCloudSurfDSNum = 0;

    // ros
    subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &Estimator::odometryHandler, this, ros::TransportHints().tcpNoDelay());
    pubOdom = nh.advertise<nav_msgs::Odometry>(odomTopic, 1);
    pubCloud = nh.advertise<sensor_msgs::PointCloud2>("fusion/estimator/local_cloud", 1);
    
	allocateMemory();
	resetLaserParameters();
	resetImageParameters();

	ROS_INFO("Estimator begins");
}

Estimator::~Estimator()
{
    processThread.join();
}

void Estimator::setParameter()
{
	mProcess.lock();
	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		tic[i] = TIC[i];
		ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
	}
	td = TD;
	g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

	if (!initThreadFlag)
	{
		ROS_INFO("Thread begins");
		initThreadFlag = true;
		processThread = std::thread(&Estimator::processMeasurements, this);
	}
	mProcess.unlock();
}

void Estimator::allocateMemory()
{
	laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
	tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>);
	fullCloud.reset(new pcl::PointCloud<PointType>());
	extractedCloud.reset(new pcl::PointCloud<PointType>());
	fullCloud->points.resize(N_SCAN*HORIZON_SCAN);

	cloudInfoOut.startRingIndex.assign(N_SCAN, 0);
	cloudInfoOut.endRingIndex.assign(N_SCAN, 0);
	cloudInfoOut.pointColInd.assign(N_SCAN*HORIZON_SCAN, 0);
	cloudInfoOut.pointRange.assign(N_SCAN*HORIZON_SCAN, 0);

    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    // copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    // copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    // kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    // kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCorner.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    laserCloudSurf.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
    laserCloudCornerDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
    laserCloudSurfDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
    
	resetLaserParameters();
}

void Estimator::resetImageParameters()
{
	inputImageCnt = 0;
	imgFrameCount = 0;
}

void Estimator::resetLaserParameters()
{
	laserCloudIn->clear();
	extractedCloud->clear();
	
	imuPointerCur = 0;
	firstPointFlag = true;
	odomDeskewFlag = false;
    
    rangeMat = cv::Mat(N_SCAN, HORIZON_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

	for (size_t i = 0; i < 1000; ++i)
	{
		imuR[i].setZero();
		imuP[i].setZero();
		imuTime[i] = 0.0;
	}
}

bool Estimator::IMUAvailable(const double t)
{
	printf("scanEnd: %f\n", t);
	printf("imu back: %f\n", imuBuf.back().header.stamp.toSec());
	if (!imuBuf.empty() && t <= imuBuf.back().header.stamp.toSec())
    {
        return true;
    }
	else
		return false;
}

bool Estimator::getIMUInterval(double start, double end, vector<sensor_msgs::Imu> &imu_vec)
{
	if (imuBuf.empty())
	{
		printf("No imu\n");
		return false;		
	}
	mBuf.lock();
	printf("scanStart: %f\n", start);
	printf("scanEnd: %f\n", end);
	printf("imuBuf front: %f\n", imuBuf.front().header.stamp.toSec());
	printf("imuBuf end: %f\n", imuBuf.back().header.stamp.toSec());
	mBuf.unlock();

	if(imuBuf.empty() || imuBuf.front().header.stamp.toSec() > start || imuBuf.back().header.stamp.toSec() < end)
	{
		printf("No IMU in between\n");
		return false;
	}

	while (imuBuf.front().header.stamp.toSec() <= start - 0.01) // ~= prev cloud's end time
	{
		mBuf.lock();
		imuBuf.pop_front();
		mBuf.unlock();
	}
	for (int i = 0; i < (int)imuBuf.size(); ++i)
	{
		if (imuBuf[i].header.stamp.toSec() <= end + 0.01)
		{
			mBuf.lock();
			imu_vec.push_back(imuBuf[i]);
			mBuf.unlock();
		}
		else
		{
			break;
		}
	}
	printf("Last imuVector %f\n", imu_vec[imu_vec.size()-1].header.stamp.toSec());
	ROS_WARN("imu interval size: %d", imu_vec.size());

	return true;
}


void Estimator::odomDeskew()
{
    cloudInfoOut.odomAvailable = false;

    while (!odomBuf.empty())
    {
        ROS_WARN("imageProjection: queue size: %d", odomBuf.size());
        if (odomBuf.front().header.stamp.toSec() < scanStartTime - 0.01)
            odomBuf.pop_front();
        else
            break;
    }

    if (odomBuf.empty())
    {
        ROS_WARN("imageProjection: odomBuf empty");
        return;
    }


    if (odomBuf.front().header.stamp.toSec() > scanStartTime)
    {
        ROS_WARN("imageProjection: odomQeue in the future timestamp");
        return;
    }

    // get start odometry at the beinning of the scan
    nav_msgs::Odometry startOdomMsg;

    for (int i = 0; i < (int)odomBuf.size(); ++i)
    {
        startOdomMsg = odomBuf[i];

        if (startOdomMsg.header.stamp.toSec() < scanStartTime)
            continue;
        else
            break;
    }

    tf::Quaternion orientation;
    tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

    double roll, pitch, yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    ROS_WARN("imageProjection: initial guess");
    // Initial guess used in mapOptimization
    cloudInfoOut.initialGuessX = startOdomMsg.pose.pose.position.x;
    cloudInfoOut.initialGuessY = startOdomMsg.pose.pose.position.y;
    cloudInfoOut.initialGuessZ = startOdomMsg.pose.pose.position.z;
    cloudInfoOut.initialGuessRoll  = roll;
    cloudInfoOut.initialGuessPitch = pitch;
    cloudInfoOut.initialGuessYaw   = yaw;

    cloudInfoOut.odomAvailable = true;

    // get end odometry at the end of the scan
    odomDeskewFlag = false;

    if (odomBuf.back().header.stamp.toSec() < scanEndTime)
        return;

    nav_msgs::Odometry endOdomMsg;

    for (int i = 0; i < (int)odomBuf.size(); ++i)
    {
        endOdomMsg = odomBuf[i];

        if (endOdomMsg.header.stamp.toSec() < scanEndTime)
            continue;
        else
            break;
    }

    if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
        return;

    Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

    float rollIncre, pitchIncre, yawIncre;
    pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

    odomDeskewFlag = true;
    ROS_WARN("imageProjection: end of odomDeskew");
}

void Estimator::imuDeskew(vector<sensor_msgs::Imu> &imuVec)
{
	cloudInfoOut.imuAvailable = false;

	if (imuVec.empty()) return;

	imuPointerCur = 0;
    imuPointerCam = 0;

    for (size_t i = 0; i < imuVec.size(); i++)
    {
        sensor_msgs::Imu this_imu = imuVec[i];
        double t = imuVec[i].header.stamp.toSec();

        if (t <= scanStartTime)
            Utility::imuRPY2rosRPY(&this_imu, &cloudInfoOut.imuRollInit, &cloudInfoOut.imuPitchInit, &cloudInfoOut.imuYawInit);

        if (t > scanEndTime + 0.01)
            break;

        if (imuPointerCur == 0)
        {
            imuR[imuPointerCur] = Vector3d(0.0, 0.0, 0.0);
            imuTime[imuPointerCur] = t;
            ++imuPointerCur;
            continue;
        }

        if (imgDeskewTime > 0 && t > imgDeskewTime)
        {
            imuPointerCam = imuPointerCur;
        }

        double angular_x, angular_y, angular_z;
        Utility::imuAngular2rosAngular(&this_imu, &angular_x, &angular_y, &angular_z);

        double dt = t - imuTime[imuPointerCur-1];
        double imuRotX = imuR[imuPointerCur-1].x() + angular_x * dt;
        double imuRotY = imuR[imuPointerCur-1].y() + angular_y * dt;
        double imuRotZ = imuR[imuPointerCur-1].z() + angular_z * dt;
        imuR[imuPointerCur] = Vector3d(imuRotX, imuRotY, imuRotZ);
        imuTime[imuPointerCur] = t;

        ++imuPointerCur;
    }
    
	--imuPointerCur;

    if (imuPointerCam <= 0) 
        imageDeskewFlag = false;
    else 
        imageDeskewFlag = true;

	if (imuPointerCur <= 0) 
        return;


	cloudInfoOut.imuAvailable = true;
}

void Estimator::projectPointCloud()
{

    int cloud_size = laserCloudIn->points.size();

    // Get transformation from start to camera pose
    if (imageDeskewFlag)
    {
        int prevPointer = imuPointerCam - 1;
        double ratioFront = (imgDeskewTime - imuTime[prevPointer]) / (imuTime[imuPointerCam] - imuTime[prevPointer]);
        double ratioBack = (imuTime[imuPointerCam] - imgDeskewTime) / (imuTime[imuPointerCam] - imuTime[prevPointer]);
        float rotXCur = imuR[imuPointerCam].x() * ratioFront + imuR[prevPointer].x() * ratioBack;
        float rotYCur = imuR[imuPointerCam].y() * ratioFront + imuR[prevPointer].y() * ratioBack;
        float rotZCur = imuR[imuPointerCam].z() * ratioFront + imuR[prevPointer].z() * ratioBack;
        float posXCur, posYCur, posZCur;
        double relTime = imgDeskewTime - scanStartTime;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        ROS_WARN("Image Deskewing...");
        // debug
        printf("imuPointercam: %d\n", imuPointerCam);
        printf("posX: %f, posY: %f, posZ: %f\n", posXCur, posYCur, posZCur);
        printf("rotX: %f, rotY: %f, rotZ: %f\n", rotXCur, rotYCur, rotZCur);
        
        transStart2Cam = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);

        // Transform initialGuess to Camera 
        Eigen::Affine3f transWorld2Start = pcl::getTransformation(cloudInfoOut.initialGuessX, cloudInfoOut.initialGuessY, cloudInfoOut.initialGuessZ,
                                                        cloudInfoOut.initialGuessRoll, cloudInfoOut.initialGuessPitch, cloudInfoOut.initialGuessYaw);
        Eigen::Affine3f transWorld2Cam = transWorld2Start * transStart2Cam;
        float camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw;
        pcl::getTranslationAndEulerAngles(transWorld2Cam, camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw);
        printf("camPosX: %f, camPosY: %f, camPosZ: %f\n", camPosX, camPosY, camPosZ);
        printf("camRoll: %f, camPitch: %f, camYaw: %f\n", camRoll, rotYCur, rotZCur);
        cloudInfoOut.initialGuessX = camPosX;
        cloudInfoOut.initialGuessY = camPosY;
        cloudInfoOut.initialGuessZ = camPosZ;
        cloudInfoOut.initialGuessRoll  = camRoll;
        cloudInfoOut.initialGuessPitch = camPitch;
        cloudInfoOut.initialGuessYaw   = camYaw;

        // Change timestamp for the pointcloud
        ros::Time rosTime(imgDeskewTime);
        cloudInfoOut.header.stamp = rosTime;
        cloudInfoOut.header.frame_id = imuFrame;
    }

    // range image projection
    for (int i = 0; i < cloud_size; ++i)
    {
        PointType this_point;
        this_point.x = laserCloudIn->points[i].x;
        this_point.y = laserCloudIn->points[i].y;
        this_point.z = laserCloudIn->points[i].z;
        this_point.intensity = laserCloudIn->points[i].intensity;

        float range = Utility::pointDistance(this_point);
        if (range < lidarMinRange || range > lidarMaxRange)
            continue;

        int row_idx = laserCloudIn->points[i].ring;
        if (row_idx < 0 || row_idx >= N_SCAN){
            continue;
        }

        if (row_idx % downsampleRate != 0)
            continue;

        float horizon_angle = atan2(this_point.x, this_point.y) * 180 / M_PI;

        static float ang_res_x = 360.0/float(HORIZON_SCAN);
        int col_idx = -round((horizon_angle-90.0)/ang_res_x) + HORIZON_SCAN/2;
        if (col_idx >= HORIZON_SCAN)
            col_idx -= HORIZON_SCAN;

        if (col_idx < 0 || col_idx >= HORIZON_SCAN)
            continue;

        if (rangeMat.at<float>(row_idx, col_idx) != FLT_MAX)
            continue;

        this_point = deskewPoint(&this_point, laserCloudIn->points[i].time);

        rangeMat.at<float>(row_idx, col_idx) = range;
        int index = col_idx + row_idx * HORIZON_SCAN;
        fullCloud->points[index] = this_point;
    }
}

void Estimator::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
{
    *posXCur = 0; *posYCur = 0; *posZCur = 0;

    // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

    if (cloudInfoOut.odomAvailable == false || odomDeskewFlag == false)
        return;

    float ratio = relTime / (scanEndTime - scanStartTime);

    *posXCur = ratio * odomIncreX;
    *posYCur = ratio * odomIncreY;
    *posZCur = ratio * odomIncreZ;
}

void Estimator::findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
{
    *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur)
    {
        if (pointTime < imuTime[imuPointerFront])
            break;
        ++imuPointerFront;
    }
    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) 
    {
        *rotXCur = imuR[imuPointerFront].x(); 
        *rotYCur = imuR[imuPointerFront].y();
        *rotZCur = imuR[imuPointerFront].z();
    } else {
        int imuPointerBack = imuPointerFront - 1;
        double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        *rotXCur = imuR[imuPointerFront].x() * ratioFront + imuR[imuPointerBack].x() * ratioBack;
        *rotYCur = imuR[imuPointerFront].y() * ratioFront + imuR[imuPointerBack].y() * ratioBack;
        *rotZCur = imuR[imuPointerFront].z() * ratioFront + imuR[imuPointerBack].z() * ratioBack;
    }
}


PointType Estimator::deskewPoint(PointType *point, double rel_time)
{
    if (deskewFlag==-1 || cloudInfoOut.imuAvailable == false)
        return *point;

    double point_time = scanStartTime + rel_time;

    float rotXCur, rotYCur, rotZCur;
    findRotation(point_time, &rotXCur, &rotYCur, &rotZCur);

    float posXCur, posYCur, posZCur;
    findPosition(rel_time, &posXCur, &posYCur, &posZCur);

    if (firstPointFlag == true)
    {
        transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
        firstPointFlag = false;
    }

    Eigen::Affine3f transCur = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBtw;

    if (imageDeskewFlag)
    {
        transBtw = transStart2Cam.inverse() * transCur;   // transform points to cam
        cloudInfoOut.isCloud = false;
    }
    else
    {
        transBtw = transStartInverse * transCur;     // transform points to start
        cloudInfoOut.isCloud = true;
    } 
    // printf("Point time: %f, Img time: %f\n", point_time, imgDeskewTime);
    // printf("Cur XYZ: %f, %f, %f\n", posXCur, posYCur, posZCur);
    // printf("Cur RPY: %f, %f, %f\n", rotXCur, rotYCur, rotZCur);    

    // float odomBtwX, odomBtwY, odomBtwZ, rollBtw, pitchBtw, yawBtw;
    // pcl::getTranslationAndEulerAngles(transBtw, odomBtwX, odomBtwY, odomBtwZ, rollBtw, pitchBtw, yawBtw);
    // printf("Btw XYZ: %f, %f, %f\n", odomBtwX, odomBtwY, odomBtwZ);
    // printf("Btw RPY: %f, %f, %f\n", rollBtw, pitchBtw, yawBtw);

    PointType newPoint;
    newPoint.x = transBtw(0,0) * point->x + transBtw(0,1) * point->y + transBtw(0,2) * point->z + transBtw(0,3);
    newPoint.y = transBtw(1,0) * point->x + transBtw(1,1) * point->y + transBtw(1,2) * point->z + transBtw(1,3);
    newPoint.z = transBtw(2,0) * point->x + transBtw(2,1) * point->y + transBtw(2,2) * point->z + transBtw(2,3);
    newPoint.intensity = point->intensity;

    return newPoint;
}

void Estimator::cloudExtraction()
{
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i) // vertical channels
    {
        cloudInfoOut.startRingIndex[i] = count - 1 + 5;

        for (int j = 0; j < HORIZON_SCAN; ++j) // horizontal channels
        {
            if (rangeMat.at<float>(i,j) != FLT_MAX)
            {
                // mark the points' column index for marking occlusion later
                cloudInfoOut.pointColInd[count] = j;
                // save range info
                cloudInfoOut.pointRange[count] = rangeMat.at<float>(i,j);
                // save extracted cloud
                extractedCloud->push_back(fullCloud->points[j + i*HORIZON_SCAN]);
                // size of extracted cloud
                ++count;
            }
        }
        cloudInfoOut.endRingIndex[i] = count -1 - 5;
    }

    // Publish deskewed cloud 
    cloudInfoOut.cloud_deskewed = Utility::toROSPointCloud(extractedCloud, cloudInfoOut.header.stamp, cloudInfoOut.header.frame_id);
    pubCloud.publish(cloudInfoOut.cloud_deskewed);
}


void Estimator::updateLatestOdometry()
{
    PointTypePose thisPoint = cloudKeyPoses6D->points[cloudKeyPoses6D->size()-1];
    latestXYZ = (Vector3d() << double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)).finished();
    latestRPY = (Vector3d() << double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)).finished();

    ROS_INFO("\033[1;33mLatestXYZ: %f, %f, %f \033[0m", latestXYZ.x(), latestXYZ.y(), latestXYZ.z());
    ROS_INFO("\033[1;33mLatestRPY: %f, %f, %f \033[0m", latestRPY.x(), latestRPY.y(), latestRPY.z());
}

void Estimator::processMeasurements()
{
	while (1)
	{
		if (!cloudBuf.empty())
		{
            // check time sync
            if (!imuBuf.empty() && !initCloudFlag)
            {
                printf("cloud front: %f\n", cloudBuf.front().header.stamp.toSec());
                printf("imu front: %f\n", imuBuf.front().header.stamp.toSec());
                if (cloudBuf.front().header.stamp.toSec() > imuBuf.front().header.stamp.toSec())
                {
                    initCloudFlag = true;
                }
                else
                {
                    mBuf.lock();
                    cloudBuf.pop();
                    mBuf.unlock();
                }
            }

            if (!initCloudFlag)
                continue;

            // cache pointcloud
            if (!cachePointCloud())
                continue;

            // check imu
            vector<sensor_msgs::Imu> imuVector;
            while(1)
            {
                if (IMUAvailable(scanEndTime) && getIMUInterval(scanStartTime, scanEndTime, imuVector))
                {
                    ROS_WARN("Imu available\n");
                    break;
                }
                else
                {
                    ROS_WARN("imuBuf size: %d\n", (int)imuBuf.size());
                    ROS_WARN("imuVector size: %d\n", (int)imuVector.size());
                    ROS_WARN("waiting for imu...\n");
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }

            // check camera
            imgDeskewTime = -1;
            pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> nearestFeature; 
            if (imageAvailable(scanStartTime) && getFirstImage(scanStartTime, scanEndTime, nearestFeature))
            {
                printf("Image available\n");
                imgDeskewTime = nearestFeature.first;
            }
            else
            {
                printf("No Image available; SLAM with only LIDAR\n");
            }
            
            // Deskew points
            imuDeskew(imuVector);
            
            odomDeskew();

            projectPointCloud();

            cloudExtraction();
            ROS_WARN("cloudExtraction\n");

            edgeSurfExtraction();
            // ROS_WARN("edgeSurfExtraction\n");

            resetLaserParameters();

            cloudInfoIn = std::move(cloudInfoBuf.front());	

            mBuf.lock();
            cloudInfoBuf.pop();
            mBuf.unlock();

            pcl::fromROSMsg(cloudInfoIn.cloud_corner, *laserCloudCorner);
            pcl::fromROSMsg(cloudInfoIn.cloud_surface, *laserCloudSurf);

            cout << "Edge:" << laserCloudCorner->points.size() << endl;
            cout << "Surf:" << laserCloudSurf->points.size() << endl;

            if (laserCloudCorner->points.size() == 0 && laserCloudSurf->points.size() == 0)
            {
                printf("Invalid pointcloud\n");
                continue;
            }

            // Map Optimization
            mProcess.lock();
            
            mapOptimizer.setCloudInfo(cloudInfoIn);

            static double timeLastProcessing = -1;
            if (mapOptimizer.timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
            {
                timeLastProcessing = mapOptimizer.timeLaserInfoCur;

                mapOptimizer.updateInitialGuess();

                mapOptimizer.extractSurroundingKeyFrames();

                mapOptimizer.downsampleCurrentScan();

                mapOptimizer.scan2MapOptimization();
                
                mapOptimizer.saveKeyFramesAndFactor();

                // mapOptimizer::correctPoses();
                // mapOptimizer::publishOdometry();
            }
            mProcess.unlock();
        }
		else if (!featureBuf.empty())
		{
		}
		else
		{

		}
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

void Estimator::inputCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	mBuf.lock();
	cloudBuf.push(*cloud_msg);
    printf("cloudBuf size: %d\n", (int)cloudBuf.size());
	mBuf.unlock();
}

// void Estimator::imageDeskew(const double time)
// {
    
// }

bool Estimator::imageAvailable(const double time)
{
    if (!featureBuf.empty())
    {
        printf("scanStart: %f\n", time);
        printf("first image: %f\n", featureBuf.front().first);
        printf("last image: %f\n", featureBuf.back().first);
        if (time <= featureBuf.back().first)
        {
            printf("Image available\n");
            return true;
        }
    } 
    printf("No Image to use\n");
    return false;
}

bool Estimator::getFirstImage(const double start, const double end, pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> &feature)
{
    // Pop old features
    while (featureBuf.front().first < start)
    {
        mBuf.lock();
        featureBuf.pop_front();
        mBuf.unlock();
    }

    // Get first feature
    if (!featureBuf.empty())
    {
        if (featureBuf.front().first < end)
        {
            feature = featureBuf.front();
            printf("First image at time: %f\n", feature.first);
            return true;
        }
    }
    return false;
}

void Estimator::edgeSurfExtraction()
{
	fusion_estimator::CloudInfo extractedCloudInfo;
	
	ROS_INFO("extract features");

	extractedCloudInfo = featureExtractor.extractEdgeSurfFeatures(cloudInfoOut);
	ROS_INFO("Info cloudinfoBuf");

	mBuf.lock();
	cloudInfoBuf.push(extractedCloudInfo);
	mBuf.unlock();	
}

bool Estimator::cachePointCloud()
{	
	if (cloudBuf.size() <= 2)
	{
		return false;
	}

	currentCloud = std::move(cloudBuf.front());
	
	mBuf.lock();
	cloudBuf.pop();
	mBuf.unlock();

	// convert point cloud according to sensor type 
	if (sensor == SensorType::VELODYNE)
    {
        pcl::moveFromROSMsg(currentCloud, *laserCloudIn);
    }
    else if (sensor == SensorType::OUSTER)
    {
        // Convert to Velodyne format
        pcl::moveFromROSMsg(currentCloud, *tmpOusterCloudIn);
        laserCloudIn->points.resize(tmpOusterCloudIn->size());
        laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
        for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
        {
            auto &src = tmpOusterCloudIn->points[i];
            auto &dst = laserCloudIn->points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            dst.ring = src.ring;
            dst.time = src.t * 1e-9f;
        }
    }
    else
    {
        ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
        ros::shutdown();
    }


    // check dense
    if (laserCloudIn->is_dense == false)
    {
        ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
        ros::shutdown();
    }

    // get timestamp
    cloudInfoOut.header = currentCloud.header;
    cloudInfoOut.header.frame_id = lidarFrame;
    // scanStartTime = currentCloud.first;
    scanStartTime = cloudInfoOut.header.stamp.toSec();
    scanEndTime = scanStartTime + laserCloudIn->points.back().time;

    // check ring channel
    static int ring_flag = 0;
    if (ring_flag == 0)
    {
        ring_flag = -1;
        for (int i = 0; i < (int)currentCloud.fields.size(); ++i)
        {
            if (currentCloud.fields[i].name == "ring")
            {
                ring_flag = 1;
                break;
            }
        }
        if (ring_flag == -1)
        {
            ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
            ros::shutdown();
        }
    }

    // check point time
    if (deskewFlag == 0)
    {
        deskewFlag = -1;
        for (auto &field : currentCloud.fields)
        {
            if (field.name == "time" || field.name == "t")
            {
                deskewFlag = 1;
                break;
            }
        }
        if (deskewFlag == -1)
            ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
    }
    return true;
}

void Estimator::inputImage(double t, const cv::Mat &img)
{
    // image time error should be fixed
    prevImgTime = t;
	inputImageCnt++;
	map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

	featureFrame = featureTracker.trackImage(t, img);

	// if (inputImageCnt % 2 == 0) // 바꿔도 됨 
	// {
		mBuf.lock();
		featureBuf.push_back(make_pair(t, featureFrame));
		mBuf.unlock();
	// }
}

void Estimator::inputIMU(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBuf.lock();
    imuBuf.push_back(*imu_msg);
    mBuf.unlock();
}

void Estimator::odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
{
	mBuf.lock();
	odomBuf.push_back(*odomMsg);
    printf("OdomBuf size: %d\n", odomBuf.size());
	mBuf.unlock();
}

gtsam::Pose3 Estimator::pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

gtsam::Pose3 Estimator::trans2gtsamPose(Eigen::Vector3d XYZ, Eigen::Vector3d RPY)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(RPY.x(), RPY.y(), RPY.z()), 
                              gtsam::Point3(XYZ.x(), XYZ.y(), XYZ.z()));
}
