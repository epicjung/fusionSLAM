#include "utility.h"

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;
    std::mutex imgLock;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    ros::Subscriber subUV;
    std::deque<pair<double, map<uint, Eigen::Vector2d>>> uvQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;


    sensor_fusion::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    // Image deskewing
    pair<double, map<uint, Eigen::Vector2d>> deskewImage;
    double imgDeskewTime;
    bool imgDeskewFlag;
    int imuPointerImg;
    Eigen::Affine3f transStart2Cam;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(cloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        subUV         = nh.subscribe<sensor_msgs::PointCloud>("tracked_feature", 100, &ImageProjection::uvHandler, this, ros::TransportHints().tcpNoDelay());
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<sensor_fusion::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*HORIZON_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*HORIZON_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*HORIZON_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, HORIZON_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;
        imgDeskewFlag = false;
        imgDeskewTime = -1;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
            
    void uvHandler(const sensor_msgs::PointCloudConstPtr& msg)
    {
        lock_guard<mutex> lock3(imgLock);
        // skip first keyframes (to-do)

        // feature measurements
        std::vector<uint> ids;
        std::vector<Eigen::Vector2d> uvs;

        // loop through features and append (to-do: vector to queue)
        map<uint, Eigen::Vector2d> feature;
        for(size_t i = 0; i < msg->points.size(); ++i)
        {
            Eigen::Vector2d uv;
            int id = msg->channels[0].values[i];
            uv << msg->points[i].x, msg->points[i].y;
            feature[uint(id)] = uv;
        }
        uvQueue.push_back(make_pair(ROS_TIME(msg), feature));
    }
       

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {

        if (!cachePointCloud(laserCloudMsg))
        {
            return;
        }

        if (!getFirstImage(timeScanCur, timeScanEnd))
        { 
            printf("Image Not available\n");
        }

        if (!deskewInfo())
        {
            ROS_WARN("imageProjection: deskew failure");
            return;
        }

        // temp
        imgDeskewFlag = imuPointerImg > 0 ? true : false; 
        
        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // abandon pointcloud if its oldest message came earlier than that of imu
        if (!imuQueue.empty())
        {
            printf("cloud front: %f\n", cloudQueue.front().header.stamp.toSec());
            printf("imu front: %f\n", imuQueue.front().header.stamp.toSec());
            if (ROS_TIME(&cloudQueue.front()) < ROS_TIME(&imuQueue.front()))
            {
                cloudQueue.pop_front();
                return false;
            }
        } else {
            return false;
        }

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
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

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        printf("timeScanCur: %f\n", timeScanCur);
        printf("timeScanEnd: %f\n", timeScanEnd);

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
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

    bool getFirstImage(double start, double end)
    {
        lock_guard<mutex> lock3(imgLock);
        
        // pop old features
        while (uvQueue.front().first < start)
        {
            uvQueue.pop_front();
        }

        // get first feature
        if (!uvQueue.empty())
        {
            if (uvQueue.front().first < end)
            {
                imgDeskewTime = uvQueue.front().first;
                deskewImage = uvQueue.front();
                printf("First image at time: %f\n", uvQueue.front().first);
                return true;
            }
        }
        imgDeskewTime = -1;
        return false;
    }

    //imuRotX, imuRotY, imuRotZ, initial guess (startOdomMsg), endOdomMsg, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre 계산 
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);
        std::lock_guard<std::mutex> lock3(imgLock);
        // make sure IMU data available for the scan
        // imuQueue.front()가 scan 시작보다 이 전에 들어온 데이터여야 하며, back()이 scan 끝났을 때보다 이 후에 들어오는 데이터여야 deskew를 진행한다.
        // 예) imu1 imu2 timeScanCur imu3 imu4 imu5 timeScanEnd imu6 imu7
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo(); // angular velocity를 사용하여 rpy에 대한 deskewing

        odomDeskewInfo(); // odom을 사용하여 translation에 대한 deskewing

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        ROS_WARN("imuQueue size: %d", imuQueue.size());

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        imuPointerImg = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // timeScanCur보다 이전이지만 제일 가까운 Imu 값의 orientation을 현재 lidar의 init orientation으로 놓는다. 
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            if (imgDeskewTime > 0 && currentImuTime > imgDeskewTime)
            {
                imuPointerImg = imuPointerCur;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            ROS_WARN("imageProjection: queue size: %d", odomQueue.size());
            printf("odomQueue: %f, scan: %f\n", odomQueue.front().header.stamp.toSec(), timeScanCur);
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
        {
            ROS_WARN("imageProjection: odomQueue empty");
            return;
        }

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
        {
            ROS_WARN("imageProjection: odomQeue in the future timestamp");
            return;
        }

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
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
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
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

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) // [imu1, imu2, imu3, imu4(imuPointerCur), pointTime)] or [pointTime, imu1(imuPointerCur)]
        {
            // scanStart로부터 pointTime까지의 relative pose
            *rotXCur = imuRotX[imuPointerFront]; 
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else { // [imu1, imu2, imu3 (imuPointerBack), pointTime, imu4(imuPointerFront), imu5(imuPointerCur)]
            // imuPointerBAck, pointTime, imuPointerFront: linear interpolation
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
            return;

        // ScanStart부터 ScanEnd까지의 odomIncremental을 통한 pointTime까지의 XYZ Linear interpolation
        float ratio = relTime / (timeScanEnd - timeScanCur);

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        // IMU로 보정한 현재 PointTime에서의 relative rotation and position w.r.t timeScanCur에서의 odometry 

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }
        
        Eigen::Affine3f transCur = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt;
        
        if (imgDeskewFlag) // transform points to camera
        {
            transBt = transStart2Cam.inverse() * transCur;
            cloudInfo.isCloud = false;
        } else { // transform poitns to start
            transBt = transStartInverse * transCur;
            cloudInfo.isCloud = true;
        }

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        // get transformation from timeScanCur to camera pose
        if (imgDeskewFlag)
        {
            int prevPointer = imuPointerImg - 1;
            double ratioFront = (imgDeskewTime - imuTime[prevPointer]) / (imuTime[imuPointerImg] - imuTime[prevPointer]);
            double ratioBack = (imuTime[imuPointerImg] - imgDeskewTime) / (imuTime[imuPointerImg] - imuTime[prevPointer]);
            float rotXCur = imuRotX[imuPointerImg] * ratioFront + imuRotX[prevPointer] * ratioBack;
            float rotYCur = imuRotY[imuPointerImg] * ratioFront + imuRotY[prevPointer] * ratioBack;
            float rotZCur = imuRotZ[imuPointerImg] * ratioFront + imuRotZ[prevPointer] * ratioBack;
            float posXCur, posYCur, posZCur;
            double relTime = imgDeskewTime - timeScanCur;
            findPosition(relTime, &posXCur, &posYCur, &posZCur);

            ROS_WARN("Image Deskewing...");
            // debug
            printf("imuPointerImg: %d\n", imuPointerImg);
            printf("posX: %f, posY: %f, posZ: %f\n", posXCur, posYCur, posZCur);
            printf("rotX: %f, rotY: %f, rotZ: %f\n", rotXCur, rotYCur, rotZCur);
            
            transStart2Cam = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);

            // Transform initialGuess to Camera 
            Eigen::Affine3f transWorld2Start = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                            cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            Eigen::Affine3f transWorld2Cam = transWorld2Start * transStart2Cam;
            float camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw;
            pcl::getTranslationAndEulerAngles(transWorld2Cam, camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw);
            printf("camPosX: %f, camPosY: %f, camPosZ: %f\n", camPosX, camPosY, camPosZ);
            printf("camRoll: %f, camPitch: %f, camYaw: %f\n", camRoll, rotYCur, rotZCur);
            cloudInfo.initialGuessX = camPosX;
            cloudInfo.initialGuessY = camPosY;
            cloudInfo.initialGuessZ = camPosZ;
            cloudInfo.initialGuessRoll  = camRoll;
            cloudInfo.initialGuessPitch = camPitch;
            cloudInfo.initialGuessYaw   = camYaw;

            // Change timestamp for the pointcloud
            ros::Time rosTime(imgDeskewTime);
            cloudInfo.header.stamp = rosTime;
            cloudInfo.header.frame_id = imuFrame;
        }

        int cloudSize = laserCloudIn->points.size();

        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN){
                continue;
            }

            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            static float ang_res_x = 360.0/float(HORIZON_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + HORIZON_SCAN/2;
            if (columnIdn >= HORIZON_SCAN)
                columnIdn -= HORIZON_SCAN;

            if (columnIdn < 0 || columnIdn >= HORIZON_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * HORIZON_SCAN;
            fullCloud->points[index] = thisPoint;
        }
        // fullCloud에 start로 땡긴 Point들의 위치를 갖고 있음 
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i) // vertical channels
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < HORIZON_SCAN; ++j) // horizontal channels
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*HORIZON_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo); // publish deskewed pointcloud
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_fusion");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    
    return 0;
}
