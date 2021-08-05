#include "utility.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;
    std::mutex uvLock;
    std::mutex imgLock;
    std::mutex lidarLock;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    // Test publisher
    ros::Publisher pubExtractedCloudCam;
    ros::Publisher pubProjectedImage;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    ros::Subscriber subUV;
    // std::deque<pair<double, map<uint, Eigen::Vector2d>>> uvQueue;
    std::deque<sensor_msgs::PointCloud> uvQueue;

    ros::Subscriber subImage;
    std::deque<pair<double, sensor_msgs::Image>> imgQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    std::deque<pcl::PointCloud<PointType>> globalCloudQueue;
    std::deque<double> timeQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;
    Eigen::Affine3f affineW2C;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;
    pcl::PointCloud<PointType>::Ptr   accumulatedCloud;

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
    sensor_msgs::PointCloud pointFeature;
    double imgDeskewTime;
    bool imgDeskewFlag;
    int imuPointerImg;
    Eigen::Affine3f affineStart2Point;

    // camera related
    vector<camodocal::CameraPtr> m_camera;


    // test
    map<int, vector<int>> threeNearest;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(cloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        subUV         = nh.subscribe<sensor_msgs::PointCloud>("/fusion/visual/tracked_feature", 1000, &ImageProjection::uvHandler, this, ros::TransportHints().tcpNoDelay());
        subImage      = nh.subscribe<sensor_msgs::Image>(imgTopic, 1000, &ImageProjection::imgHandler, this, ros::TransportHints().tcpNoDelay());
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("fusion/deskew/cloud_deskewed", 1);
        pubExtractedCloudCam = nh.advertise<sensor_msgs::PointCloud2>("fusion/deskew/cam_deskewed", 1);
        pubProjectedImage = nh.advertise<sensor_msgs::Image>("projected_img", 1);
        pubLaserCloudInfo = nh.advertise<sensor_fusion::cloud_info> ("fusion/deskew/cloud_info", 1);

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
        accumulatedCloud.reset(new pcl::PointCloud<PointType>());

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
        // sensor_msgs::Imu thisImu = *imuMsg;
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

    void transformOdometry(const nav_msgs::Odometry src, nav_msgs::Odometry &dest)
    {
        
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void imgHandler(const sensor_msgs::ImageConstPtr &imgMsg)
    {
        lock_guard<mutex> lock4(imgLock);
        imgQueue.push_back(make_pair(ROS_TIME(imgMsg), *imgMsg)); 
    }

    void uvHandler(const sensor_msgs::PointCloudConstPtr &msg)
    {
        lock_guard<mutex> lock3(uvLock);
        printf("incoming uv time past: %f\n", msg->header.stamp.toSec());
        uvQueue.push_back(*msg);
    }
       

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        cloudInfo.isCloud = true; // default

        if (!cachePointCloud(laserCloudMsg))
        {
            return;
        }

        if (!getFirstImage(timeScanCur, timeScanEnd))
        { 
            printf ("\033[31;1m Image Not Available \033[0m\n");
        }
        else
        {
            printf ("\033[32;1m Image Available \033[0m\n");
            printf("imgDeskewTime: %f\n", imgDeskewTime);
        }

        if (!deskewInfo())
        {
            ROS_WARN("imageProjection: deskew failure");
            return;
        }

        projectPointCloud();

        cloudExtraction();

        stackPointCloud();

        if (imgDeskewFlag)
            associatePointFeature();

        publishClouds();

        // visualizeProjection();

        resetParameters();
    }

    void stackPointCloud()
    {
        TicToc tictoc;

        lock_guard<mutex> lock(lidarLock);

        static int lidarCount = -1;
        // TO-DO: this should modified to stack the pointcloud even if there is no image to deskew
        if (cloudInfo.isCloud || ++lidarCount % (lidarSkip+1) != 0) 
            return;

        // 0. Filter cloud (downsample and keep only points in camera view)
        pcl::PointCloud<PointType>::Ptr extractedCloudDS(new pcl::PointCloud<PointType>()); 
        static pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.setInputCloud(extractedCloud);
        downSizeFilter.filter(*extractedCloudDS);
        *extractedCloud = *extractedCloudDS;
        
        pcl::PointCloud<PointType>::Ptr extractedCloudFilter(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int) extractedCloud->size(); ++i)
        {
            PointType p = extractedCloud->points[i];
            if (p.z > 0.0 && abs(p.y / p.z) < 10.0 && abs(p.x / p.z) < 10.0)
            {
                extractedCloudFilter->push_back(p);
            }
        }
        *extractedCloud = *extractedCloudFilter;
        
        // 1. Transform pointcloud to global 
        pcl::PointCloud<PointType>::Ptr globalCloud(new pcl::PointCloud<PointType>());
        Eigen::Affine3f affineW2C = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                    cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
        pcl::transformPointCloud(*extractedCloud, *globalCloud, affineW2C);

        ROS_WARN("Transform pointcloud global time: %fms", tictoc.toc());

        // 2. Queue the global cloud and time
        globalCloudQueue.push_back(*globalCloud);
        timeQueue.push_back(imgDeskewTime);

        // 3. Only accumulate recent clouds
        while (!timeQueue.empty())
        {
            if (imgDeskewTime - timeQueue.front() > stackPointTime)
            {
                globalCloudQueue.pop_front();
                timeQueue.pop_front();
            } else {
                break;
            }
        }
        accumulatedCloud->clear();
        int count = 0;
        for (int i = 0; i < (int)globalCloudQueue.size(); ++i)
        {
            *accumulatedCloud += globalCloudQueue[i];
            count ++;
        }    

        ROS_WARN("Add pointcloud time: %fms", tictoc.toc());

        // // 4. Downsample accumulated global cloud
        // pcl::PointCloud<PointType>::Ptr accumulatedCloudDS(new pcl::PointCloud<PointType>());
        // static pcl::VoxelGrid<PointType> downSizeFilter;
        // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        // downSizeFilter.setInputCloud(accumulatedCloud);
        // downSizeFilter.filter(*accumulatedCloudDS);
        // *accumulatedCloud = *accumulatedCloudDS;
        ROS_WARN("Accumulated %d set of %d pointcloud time: %fms", count, accumulatedCloud->size(), tictoc.toc());
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
        lock_guard<mutex> lock3(uvLock);
        
        // pop old features
        while (!uvQueue.empty())
        {
            printf("Front feature size: %d\n", uvQueue.front().points.size());
            if (ROS_TIME(&uvQueue.front()) > start)
            {
                if (ROS_TIME(&uvQueue.front()) < end)
                {
                    pointFeature = uvQueue.front();
                    imgDeskewTime = ROS_TIME(&pointFeature);
                    printf("First image at time: %f\n", imgDeskewTime);
                    return true;
                }
                else
                {
                    imgDeskewTime = -1;
                    return false;
                }
            }
            else
            {
                uvQueue.pop_front();
            }
        }
        printf("Empty uv queue\n");
        return false;
    }

    //imuRotX, imuRotY, imuRotZ, initial guess (startOdomMsg), endOdomMsg, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre 계산 
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);
        std::lock_guard<std::mutex> lock3(uvLock);
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

        imgDeskewFlag = (imuPointerImg > 0 && cloudInfo.odomAvailable) ? true : false; // imgDeskew only when odom is available

        if (imgDeskewFlag)
            printf ("\033[32;1m Image Deskew \033[0m\n");
        else
            printf ("\033[31;1m No Image Deskew \033[0m\n");

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

            if (imgDeskewTime > 0)
            {
                if (currentImuTime <= imgDeskewTime)
                    imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
                else
                {
                    if (imuPointerImg == 0)
                        imuPointerImg = imuPointerCur;
                }
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation (imu에서 lidar로 transform 필요)
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
        printf("imuPointerImage: %d, imuPointerCur: %d\n", imuPointerImg, imuPointerCur);
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            // ROS_WARN("imageProjection: queue size: %d", odomQueue.size());
            // printf("odomQueue: %f, scan: %f\n", odomQueue.front().header.stamp.toSec(), timeScanCur);
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
            transBt = affineL2C.inverse() * affineStart2Point.inverse() * transCur;
        } else { // transform poitns to start
            transBt = transStartInverse * transCur;
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
            // find rotation
            int prevPointer = imuPointerImg - 1;
            double ratioFront = (imgDeskewTime - imuTime[prevPointer]) / (imuTime[imuPointerImg] - imuTime[prevPointer]);
            double ratioBack = (imuTime[imuPointerImg] - imgDeskewTime) / (imuTime[imuPointerImg] - imuTime[prevPointer]);
            float rotXCur = imuRotX[imuPointerImg] * ratioFront + imuRotX[prevPointer] * ratioBack;
            float rotYCur = imuRotY[imuPointerImg] * ratioFront + imuRotY[prevPointer] * ratioBack;
            float rotZCur = imuRotZ[imuPointerImg] * ratioFront + imuRotZ[prevPointer] * ratioBack;
            float posXCur, posYCur, posZCur;
            double relTime = imgDeskewTime - timeScanCur;

            //find position
            findPosition(relTime, &posXCur, &posYCur, &posZCur);

            ROS_WARN("Image Deskewing...");
            // debug
            printf("imuPointerImg: %d\n", imuPointerImg);
            printf("posX: %f, posY: %f, posZ: %f\n", posXCur, posYCur, posZCur);
            printf("rotX: %f, rotY: %f, rotZ: %f\n", rotXCur, rotYCur, rotZCur);
            
            affineStart2Point = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
            cout << affineStart2Point.matrix() << endl;
            // Transform initialGuess to Camera 
            Eigen::Affine3f affineWorld2Start = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                            cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            
            affineW2C = affineWorld2Start * affineStart2Point * affineL2C;
                    
            ROS_WARN("imageProjection: initial guess");
            float initX, initY, initZ, initRoll, initPitch, initYaw;
            pcl::getTranslationAndEulerAngles(affineWorld2Start, initX, initY, initZ, initRoll, initPitch, initYaw);
            printf("ScanStartTime: %f\n", timeScanCur);
            printf("initX: %f, initY: %f, initZ: %f\n", initX, initY, initZ);
            printf("initRoll: %f, initPitch: %f, initYaw: %f\n", initRoll, initPitch, initYaw);            

            float camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw;
            pcl::getTranslationAndEulerAngles(affineW2C, camPosX, camPosY, camPosZ, camRoll, camPitch, camYaw);
            cout << affineW2C.matrix() << endl;
            printf("camPosX: %f, camPosY: %f, camPosZ: %f\n", camPosX, camPosY, camPosZ);
            printf("camRoll: %f, camPitch: %f, camYaw: %f\n", camRoll, camPitch, camYaw);
            
            // tmp
            Eigen::Affine3f imuInitRPY = pcl::getTransformation(0.0, 0.0, 0.0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            imuInitRPY = imuInitRPY * affineL2C;
            float x,y,z;
            pcl::getTranslationAndEulerAngles(imuInitRPY, x, y, z, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);

            // Send transform
            // Change timestamp for the pointcloud
            ros::Time rosTime(imgDeskewTime);
            cloudInfo.header.stamp = rosTime;
            cloudInfo.header.frame_id = camFrame;
            cloudInfo.initialGuessX = camPosX;
            cloudInfo.initialGuessY = camPosY;
            cloudInfo.initialGuessZ = camPosZ;
            cloudInfo.initialGuessRoll  = camRoll;
            cloudInfo.initialGuessPitch = camPitch;
            cloudInfo.initialGuessYaw   = camYaw;
            cloudInfo.isCloud = false;
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

    void associatePointFeature()
    {
        TicToc tictoc;

        sensor_msgs::ChannelFloat32 depths;
        depths.name = "depth";
        depths.values.resize(pointFeature.points.size(), -1); // -1 for initial depth

        // TO-DO: cloudInfo LiDAR로 변경 필요
        if (accumulatedCloud->size() == 0)
            return;

        // 1. Transform pointcloud to camera frame
        pcl::PointCloud<PointType>::Ptr localCloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*accumulatedCloud, *localCloud, affineW2C.inverse());
        
        // 2. Project depth cloud on the range image and filter points in the same region
        int numBins = 360; 
        vector<vector<PointType>> pointArray;
        pointArray.resize(numBins);
        for (int i = 0; i < numBins; ++i)
            pointArray[i].resize(numBins);
        float angRes = 180.0 / float(numBins); // cover only -90~90 FOV of the lidar 
        cv::Mat rangeImage = cv::Mat(numBins, numBins, CV_32F, cv::Scalar::all(FLT_MAX));

        for (int i = 0; i < (int)localCloud->size(); ++i)
        {
            PointType p = localCloud->points[i];
            if (p.z > 0.0 && abs(p.y / p.z) < 10.0 && abs(p.x / p.z) < 10.0)
            {
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

        // TO-DO: cloudInfo LiDAR로 변경 필요
        if (localCloudSphere->size() < 10) 
            return;

        // 5. Project 2D point feature to unit sphere
        pcl::PointCloud<PointType>::Ptr pointSphere(new pcl::PointCloud<PointType>());        
        for (int i = 0; i < (int)pointFeature.points.size(); ++i)
        {
            Eigen::Vector3f thisFeature(pointFeature.points[i].x, pointFeature.points[i].y, 1.0);
            thisFeature.normalize();
            PointType p;
            p.x = thisFeature(0);
            p.y = thisFeature(1);
            p.z = thisFeature(2);
            p.intensity = -1; // depth
            pointSphere->push_back(p);
        }

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
                    depths.values[i] = pointSphere->points[i].intensity;
                    pointFeature.channels[1].values[i] = pointSphere->points[i].x;
                    pointFeature.channels[2].values[i] = pointSphere->points[i].y;
                    pointFeature.channels[3].values[i] = pointSphere->points[i].z;
                }
            }
        }

        publishCloud(&pubExtractedCloudCam, threePoints, cloudInfo.header.stamp, camFrame);

        visualizeAssociatedPoints(depths);

        cloudInfo.point_feature = pointFeature;

        /* DEBUG */
        ROS_WARN("Associate point feature time: %fms\n", tictoc.toc());
        printf("PointFeature size: %d\n", pointFeature.points.size());
        int count = 0;
        for (int i = 0; i < (int)depths.values.size(); ++i)
        {   
            if (depths.values[i] >= 0)
            {
                count++;
            }
        }
        printf("AssociatedFeature size: %d\n", count);
    }

    void visualizeAssociatedPoints(sensor_msgs::ChannelFloat32 depths)
    {
          // pop old images
        cv_bridge::CvImagePtr cv_ptr;
        bool isFound = false;
        {
            lock_guard<mutex> lock4(imgLock);
            
            while (!imgQueue.empty())
            {
                if (imgQueue.front().first < imgDeskewTime)
                {
                    imgQueue.pop_front();
                }
                else if (imgQueue.front().first == imgDeskewTime)
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
            // project 3d to image plane
            pcl::PointCloud<PointType>::Ptr localCloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*accumulatedCloud, *localCloud, affineW2C.inverse());
            for (auto& point : localCloud->points)
            {
                if (point.z > 0)
                {
                    Vector3d spacePoint(point.x, point.y, point.z);
                    float maxVal = 20.0;
                    int red = min(255, (int)(255 * abs((point.z - maxVal) / maxVal)));
                    int green = min(255, (int)(255 * (1 - abs((point.z - maxVal) / maxVal ))));
                    Vector2d imagePoint;
                    m_camera[0]->spaceToPlane(spacePoint, imagePoint);
                    if (imagePoint.x() >= 0 && imagePoint.x() < m_camera[0]->imageWidth() && imagePoint.y() >=0 && imagePoint.y() < m_camera[0]->imageHeight())
                    {
                        // printf("3d: %f, %f, %f\n", point.x, point.y, point.z);
                        // printf("uv: %f, %f\n", imagePoint.x(), imagePoint.y());
                        // printf("red: %d, green %d\n", red, green);
                        cv::Point2f uv(imagePoint.x(), imagePoint.y());
                        cv::circle(cv_ptr->image, uv, 0.5, cv::Scalar(0, green, red), -1);
                    }
                }
            }

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
            pubProjectedImage.publish(cv_ptr->toImageMsg());
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
    
    signal(SIGINT, signal_handle::signal_callback_handler);

    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    
    return 0;
}
