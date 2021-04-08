#include <stdio.h>
#include <queue>
#include <cv_bridge/cv_bridge.h>
#include "utility/parameters.h"
#include "estimator/estimator.h"

class Manager : public ParamServer
{
    private:
        ros::Subscriber subImu;
        ros::Subscriber subCloud;
        ros::Subscriber subImage;

    public:

        Estimator estimator;

        std::mutex mBuf;

        Manager()
        {
            subImu = nh.subscribe(imuTopic, 2000, &Manager::imuCallback, this, ros::TransportHints().tcpNoDelay());
            subImage = nh.subscribe(imgTopic, 100, &Manager::imgCallback, this, ros::TransportHints().tcpNoDelay());
            subCloud = nh.subscribe(cloudTopic, 5, &Manager::cloudCallback, this, ros::TransportHints().tcpNoDelay());
        }

        ~Manager(){}

        cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
        {
            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = img_msg->header;
                img.height = img_msg->height;
                img.width = img_msg->width;
                img.is_bigendian = img_msg->is_bigendian;
                img.step = img_msg->step;
                img.data = img_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat img = ptr->image.clone();
            return img;
        }

        void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
        {
            // double dx = imu_msg->linear_acceleration.x;
            // double dy = imu_msg->linear_acceleration.y;
            // double dz = imu_msg->linear_acceleration.z;
            // double rx = imu_msg->angular_velocity.x;
            // double ry = imu_msg->angular_velocity.y;
            // double rz = imu_msg->angular_velocity.z;
            // Eigen::Vector3d acc(dx, dy, dz);
            // Eigen::Vector3d gyr(rx, ry, rz);
            estimator.inputIMU(imu_msg);
            return;
        }

        void imgCallback(const sensor_msgs::ImageConstPtr &img_msg)
        {
            cv::Mat img;
            double t = 0.0;
            t = img_msg->header.stamp.toSec();
            img = getImageFromMsg(img_msg);
            if (!img.empty())
                estimator.inputImage(t, img);
            return;
        }

        void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
        {
            estimator.inputCloud(cloud_msg);
            return;
        }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "fusion_estimator");

    Manager manager;

    manager.estimator.setParameter();

    ROS_INFO("\033[1;32m----> Estimator node started.\033[0m");

    // ROS_INFO("\033[1;32m----> Estimator node started.\033[0m");

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}