#include "utility.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

const int queueLength = 2000;

class FeatureManager : public ParamServer
{
private:
    mutex pointLock;
    mutex imgLock;
    mutex odomLock;
    mutex imuLock;

    deque<sensor_msgs::PointCloud> pointQueue;
    deque<pair<double, sensor_msgs::Image>> imgQueue; 
    deque<nav_msgs::Odometry> odomQueue;
    deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subPoint;       
    ros::Subscriber subCloudInfo;    
    ros::Subscriber subImage;  
    ros::Subscriber subOdom;

    sensor_msgs::PointCloud pointFeature;

    sensor_fusion::cloud_info cloudInfo;       
    
    pcl::PointCloud<PointType>::Ptr currentCloud;

    vector<camodocal::CameraPtr> m_camera;
    double imgTime;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];
    int imuPointerCur;

    float rotXCur, rotYCur, rotZCur;
    float posXCur, posYCur, posZCur;

    double timeScanStart;
    double timeScanEnd;

public: 
    FeatureManager()
    {
        subPoint        = nh.subscribe<sensor_msgs::PointCloud>("/fusion/visual/point", 1000, &FeatureManager::pointHandler, this, ros::TransportHints().tcpNoDelay());
        subCloudInfo    = nh.subscribe<sensor_fusion::cloud_info>("/fusion/deskew/cloud_info", 1000, &FeatureManager::cloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subImage        = nh.subscribe<sensor_msgs::Image>(imgTopic, 1000, &FeatureManager::imgHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom         = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &FeatureManager::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        allocateMemory();
    }  

    void allocateMemory()
    {
        currentCloud.reset(new pcl::PointCloud<PointType>());
        resetParameters();

         // get camera info
        for (size_t i = 0; i < CAM_NAMES.size(); i++)
        {
            ROS_DEBUG("reading paramerter of camera %s", CAM_NAMES[i].c_str());
            FILE *fh = fopen(CAM_NAMES[i].c_str(), "r");
            if (fh == NULL)
            {
                ROS_WARN("config_file doesn't exist");
                ROS_BREAK();
                return;
            }
            fclose(fh);

            camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_NAMES[i]);
            m_camera.push_back(camera);
        }
    }

    void resetParameters()
    {
        currentCloud->clear();
        imgTime = -1;
        imuPointerCur = 0;

        // reset queue
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        } 
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        std::lock_guard<std::mutex> lock(imuLock);
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        lock_guard<mutex> lock(odomLock);
        odomQueue.push_back(*odometryMsg);
    }

    void pointHandler(const sensor_msgs::PointCloudConstPtr &msg)
    {
        lock_guard<mutex> lock(pointLock);
        pointQueue.push_back(*msg);
    }

    void imgHandler(const sensor_msgs::ImageConstPtr &imgMsg)
    {
        lock_guard<mutex> lock(imgLock);
        imgQueue.push_back(make_pair(ROS_TIME(imgMsg), *imgMsg));
    }


    void cloudInfoHandler(const sensor_fusion::cloud_info::ConstPtr &cloudInfoMsg)
    {
        cloudInfo = *cloudInfoMsg;

        if (!cloudInfo.odomAvailable || !cloudInfo.imuAvailable) // odomAvailable
            return;
        
        timeScanStart = cloudInfo.header.stamp.toSec();
        timeScanEnd = timeScanStart + 0.1; // 10 Hz

        if (!getImageBetween())
        {
            printf ("\033[31;1m Image Not Available \033[0m\n");
            return;
        } else {
            printf ("\033[32;1m Image Available \033[0m\n");
            printf("imgDeskewTime: %f\n", imgTime);
        }

        if (!findRotation())
            return;

        

    }

    bool findPosition()
    {

    }

    bool findRotation()
    {
        while (!imuQueue.empty())
        {
            if (ROS_TIME(&imuQueue.front()) < timeScanStart - 0.01)
                imuQueue.pop_front();
            else
                break;
        }
        
        if (imuQueue.empty())
            return false;
        
        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = ROS_TIME(&thisImuMsg);

            if (currentImuTime > timeScanEnd + 0.01)
                break;
            
            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0.0;
                imuRotY[0] = 0.0;
                imuRotZ[0] = 0.0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            if (currentImuTime <= imgTime)
            {
                double angularX, angularY, angularZ;
                imuAngular2rosAngular(&thisImuMsg, &angularX, &angularY, &angularZ);
                double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
                imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angularX * timeDiff;
                imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angularY * timeDiff;
                imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angularZ * timeDiff;
                imuTime[imuPointerCur] = currentImuTime;
                ++imuPointerCur;                
            }
        }

        if (imuPointerCur > 0)
        {
            int imuPointerPrev = imuPointerCur - 1;
            double ratioFront = (imgTime - imuTime[imuPointerPrev]) / (imuTime[imuPointerCur] - imuTime[imuPointerPrev]);
            double ratioBack  = (imuTime[imuPointerCur] - imgTime) /  (imuTime[imuPointerCur] - imuTime[imuPointerPrev]);
            rotXCur = imuRotX[imuPointerCur] * ratioFront + imuRotX[imuPointerPrev] * ratioBack;
            rotYCur = imuRotY[imuPointerCur] * ratioFront + imuRotY[imuPointerPrev] * ratioBack;
            rotZCur = imuRotZ[imuPointerCur] * ratioFront + imuRotZ[imuPointerPrev] * ratioBack;
            printf("imgPointer at %d from queue size %d \n", imuPointerCur, imuQueue.size());
            ROS_WARN("rotXCur: %f, rotYCur: %f, rotZCur: %f\n", rotXCur, rotYCur, rotZCur);
            return true;
        }

        return false;
    }

    bool getImageBetween()
    {
        lock_guard<mutex> lock(pointLock);
        
        while (!pointQueue.empty())
        {
            if (ROS_TIME(&pointQueue.front()) > timeScanStart)
            {
                if (ROS_TIME(&pointQueue.front()) < timeScanEnd)
                {
                    pointFeature = pointQueue.front();
                    imgTime = ROS_TIME(&pointFeature);
                    printf("Image at time: %f\n", imgTime);
                    return true;
                }
                else
                {
                    imgTime = -1;
                    return false;
                }
            } else {
                pointQueue.pop_front();
            }
        }
        printf("Empty point queue\n");
        return false;
    }

};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_fusion");

    FeatureManager FE;

    ROS_INFO("\033[1;32m----> Feature Manager Started.\033[0m");
   
    ros::spin();

    return 0;
}