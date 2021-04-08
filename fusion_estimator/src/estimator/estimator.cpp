#include "estimator.h"

Estimator::Estimator()
{
	initThreadFlag= false;
	initSystemFlag = false;
	deskewFlag = 0;
	cloudFrameCount = 0;
    prevImgTime = -1.0;

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    laserCloudCornerFromMapDSNum = 0;
    laserCloudSurfFromMapDSNum = 0;
    laserCloudCornerDSNum = 0;
    laserCloudSurfDSNum = 0;

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

void Estimator::resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = new ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newFactorGraph;
    factorGraph = newFactorGraph;

    gtsam::Values newInitialEstimate;
    gtsam::Values initialEstimate = newInitialEstimate;
    gtsam::Values newCurrentEstimate;
    gtsam::Values isamCurrentEstimate = newCurrentEstimate;
}

void Estimator::resetImageParameters()
{
	mProcess.lock();
	inputImageCnt = 0;
	imgFrameCount = 0;
	mProcess.unlock();
}

void Estimator::resetLaserParameters()
{
	mProcess.lock();
	laserCloudIn->clear();
	extractedCloud->clear();
	
	imuPointerCur = 0;
	firstPointFlag = true;
	odomDeskewFlag = false;
	lastImuTime = -1;
    
    rangeMat = cv::Mat(N_SCAN, HORIZON_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

	for (size_t i = 0; i < 1000; ++i)
	{
		imuR[i].setZero();
		imuP[i].setZero();
		imuTime[i] = 0.0;
	}
	mProcess.unlock();
}

bool Estimator::IMUAvailable(const double t)
{
	if (!imuBuf.empty() && t <= imuBuf.back().header.stamp.toSec())
		return true;
	else
		return false;
}

bool Estimator::getIMUInterval(double start, double end, vector<sensor_msgs::Imu> &imu_vec)
{
	if(imuBuf.empty() || imuBuf.front().header.stamp.toSec() > start || imuBuf.back().header.stamp.toSec() < end)
	{
		return false;
	}

	if(end <= imuBuf.back().header.stamp.toSec())
	{
		while (imuBuf.front().header.stamp.toSec() <= start - 0.01)
		{
			imuBuf.pop();
		}

		while (imuBuf.front().header.stamp.toSec() <= end + 0.01)
		{
			imu_vec.push_back(imuBuf.front());
			imuBuf.pop();
		}
		ROS_WARN("imu interval size: %d", imu_vec.size());
	}
	else
	{
		return false;
	}
	return true;
}


void Estimator::odomDeskew()
{
    cloudInfoOut.odomAvailable = false;

    while (!odomBuf.empty())
    {
        ROS_WARN("imageProjection: queue size: %d", odomBuf.size());
        printf("odomBuf: %f, scan: %f\n", odomBuf.front().header.stamp.toSec(), scanStartTime);
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

void Estimator::imuDeskew(vector<sensor_msgs::Imu> &imu_vec)
{
	cloudInfoOut.imuAvailable = false;

	if (imu_vec.empty()) return;

	imuPointerCur = 0;

	for (size_t i = 0; i < imu_vec.size(); i++)
	{
		sensor_msgs::Imu this_imu = imu_vec[i];
		double t = imu_vec[i].header.stamp.toSec();

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

	if (imuPointerCur <= 0) return;

	cloudInfoOut.imuAvailable = true;
}


void Estimator::projectPointCloud()
{

    int cloud_size = laserCloudIn->points.size();

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

    // transform points to start
    Eigen::Affine3f transCur = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBtw = transStartInverse * transCur;

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

    // set up cloudInfo for feature extraction
    cloudInfoOut.header = cloudHeader;
    cloudInfoOut.cloud_deskewed = Utility::toROSPointCloud(extractedCloud, cloudHeader.stamp, lidarFrame);
}

void Estimator::updateInitialGuess()
{
	increOdomFront = trans2Affine3f(latestXYZ, latestRPY);

	static Eigen::Affine3f lastImuTrans;

	if (cloudKeyPoses3D->points.empty())
	{
		Vector3d initXYZ;
		Vector3d initRPY;

		initRPY = Vector3d(cloudInfoIn.imuRollInit, cloudInfoIn.imuPitchInit, cloudInfoIn.imuYawInit);
		initXYZ.setZero();

		latestXYZ = initXYZ;
		latestRPY = initRPY;

		lastImuTrans = trans2Affine3f(initXYZ, initRPY);
		return;
	}
}

void Estimator::extractSurroundingKeyFrames()
{
	if (cloudKeyPoses3D->points.empty())
		return;

	// extractNearby();
}

void Estimator::downsampleCurrentScan()
{
    // Downsample cloud from current scan
    laserCloudCornerDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCorner); 
    downSizeFilterCorner.filter(*laserCloudCornerDS); // Down-sampled corner points
    laserCloudCornerDSNum = laserCloudCornerDS->size();

    laserCloudSurfDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurf);
    downSizeFilterSurf.filter(*laserCloudSurfDS); // Down-sampled surface points
    laserCloudSurfDSNum = laserCloudSurfDS->size();	
}

void Estimator::scan2MapOptimization()
{
	if (cloudKeyPoses3D->points.empty())
		return;
}

bool Estimator::isCloudKeyframe()
{
	if (cloudKeyPoses3D->points.empty())
		return true;
}

void Estimator::addOdomFactor()
{
	if (cloudKeyPoses3D->points.empty())
	{
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        factorGraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(latestXYZ, latestRPY), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(latestXYZ, latestRPY));
	}
}

void Estimator::optimize()
{
	optimizer->update(factorGraph, initialEstimate);
	optimizer->update();
	factorGraph.resize(0);
	initialEstimate.clear();

	// save key poses
	PointType thisPose3D;
	Pose3 latestEstimate;
}

void Estimator::saveKeyframe(int type)
{
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

	isamCurrentEstimate = optimizer->calculateEstimate();
	latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

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
    thisPose6D.time = timeCloudInfoIn;
    cloudKeyPoses6D->push_back(thisPose6D); //cloudKeyPoses6D has xyz and rotations

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = optimizer->marginalCovariance(isamCurrentEstimate.size()-1);

    latestXYZ = (Vector3d() << latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z()).finished();
    latestRPY = (Vector3d() << latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw()).finished();

    ROS_INFO("\033[1;33mLatestXYZ: %f, %f, %f \033[0m", latestXYZ.x(), latestXYZ.y(), latestXYZ.z());
    ROS_INFO("\033[1;33mLatestRPY: %f, %f, %f \033[0m", latestRPY.x(), latestRPY.y(), latestRPY.z());



    if (type == 1) // pointcloud pose
    {
	    // save all the received edge and surf points
	    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
	    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
	    pcl::copyPointCloud(*laserCloudCornerDS,  *thisCornerKeyFrame);
	    pcl::copyPointCloud(*laserCloudSurfDS,    *thisSurfKeyFrame);
	    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
	    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    }
}

void Estimator::publishOdometry()
{

}

void Estimator::processMeasurements()
{
	while (1)
	{
		if (!cloudInfoBuf.empty())
		{
			if (imgFrameCount == WINDOW_SIZE)
			{

			}
			else
			{
				// printf("Not enough image keyframes\n");

				// if (!initSystemFlag)
				// {
					// resetOptimization();

					cloudInfoIn = std::move(cloudInfoBuf.front());	

					timeCloudInfoIn = cloudInfoIn.header.stamp.toSec();

					pcl::fromROSMsg(cloudInfoIn.cloud_corner, *laserCloudCorner);
					pcl::fromROSMsg(cloudInfoIn.cloud_surface, *laserCloudSurf);

					mBuf.lock();
					cloudInfoBuf.pop();
					mBuf.unlock();

					mProcess.lock();

					static double timeLastProcessing = -1;
					if (timeCloudInfoIn - timeLastProcessing >= mappingProcessInterval)
					{
						timeLastProcessing = timeCloudInfoIn;

						updateInitialGuess();

						extractSurroundingKeyFrames();

						downsampleCurrentScan();

						scan2MapOptimization();

						if (isCloudKeyframe())
						{
							addOdomFactor();

							optimize();

							saveKeyframe(1);
						}

					}

					mProcess.unlock();


					// printf("initImuVector size: %d\n", (int)initImuVector.size());

					// initSystemFlag = true;
				// }
			}
		}
		else if (!featureBuf.empty())
		{

		}
		else
		{
			// printf("No feature input yet.\n");
		}
	}
}

void Estimator::inputCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	mBuf.lock();
	// // solution to error 
	// if (!cloudBuf.empty())
	// {
	// 	if(cloudBuf.back().header.stamp.toSec() > cloud_msg->header.stamp.toSec())
	// 	{
	// 		while (!cloudBuf.empty())
	// 			cloudBuf.pop();
	// 	}
	// }

	cloudBuf.push(*cloud_msg);
	mBuf.unlock();

	ROS_WARN("cloudBuf size: %d", (int)cloudBuf.size());

	// cache pointcloud
	if (!cachePointCloud())
	{
		return;
	}

    vector<sensor_msgs::Imu> imuVector;
	// check imu
	while(1)
	{
        mBuf.lock();
		if (IMUAvailable(scanEndTime) && getIMUInterval(scanStartTime, scanEndTime, imuVector))
		{
			ROS_WARN("Imu available");
			break;
		}
		else
		{
            ROS_WARN("imuBuf size: %d", (int)imuBuf.size());
            ROS_WARN("imuVector size: %d", (int)imuVector.size());
			ROS_INFO("waiting for imu...");
			std::chrono::milliseconds dura(5);
			std::this_thread::sleep_for(dura);
		}
        mBuf.unlock();
	}

	imuDeskew(imuVector);

	odomDeskew();
	ROS_INFO("deskew");

	projectPointCloud();
	ROS_INFO("projectPointCloud");

	cloudExtraction();
	ROS_INFO("cloudExtraction");

	edgeSurfExtraction();
	ROS_INFO("edgeSurfExtraction");

	resetLaserParameters();
}

void Estimator::edgeSurfExtraction()
{
	fusion_estimator::CloudInfo extractedCloudInfo;
	// pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
	// pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
	// laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
	// laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());

	extractedCloudInfo = featureExtractor.extractEdgeSurfFeatures(cloudInfoOut);
    // printf("Edge size: %d\n", (int)laserCloudCornerLast->points.size());
    // printf("Surf size: %d\n", (int)laserCloudSurfLast->points.size());
	
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
    cloudHeader = currentCloud.header;
    // scanStartTime = currentCloud.first;
    scanStartTime = cloudHeader.stamp.toSec();
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

	if (inputImageCnt % 2 == 0)
	{
		mBuf.lock();
		featureBuf.push(make_pair(t, featureFrame));
		mBuf.unlock();
	}
}

void Estimator::inputIMU(const sensor_msgs::ImuConstPtr &imu_msg)
{

    mBuf.lock();
    // // check validity
    // if (!imuBuf.empty())
    // {
    //     if(imuBuf.back().header.stamp.toSec() > imu_msg->header.stamp.toSec())
    //     {
    //         printf("Invalid imu measurements. Pop all\n");
    //         while (!imuBuf.empty())
    //             imuBuf.pop();
    //     }
    // }

    imuBuf.push(*imu_msg);
    mBuf.unlock();
}

void Estimator::inputOdom(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
	mBuf.lock();
	odomBuf.push_back(*odom_msg);
	mBuf.unlock();
}

// void Estimator::inputIMU(double t, const Vector3d &acc, const Vector3d &gyr)
// {
// 	mBuf.lock();
// 	acc_buf.push(make_pair(t, acc));
// 	gyr_buf.push(make_pair(t, gyr));
// 	mBuf.unlock();
// }

Affine3f Estimator::trans2Affine3f(Vector3d XYZ, Vector3d RPY)
{
    return pcl::getTransformation(XYZ.x(), XYZ.y(), XYZ.z(), RPY.x(), RPY.y(), RPY.z());
}

gtsam::Pose3 Estimator::trans2gtsamPose(Eigen::Vector3d XYZ, Eigen::Vector3d RPY)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(RPY.x(), RPY.y(), RPY.z()), 
                              gtsam::Point3(XYZ.x(), XYZ.y(), XYZ.z()));
}
