#include <stdio.h>
#include <queue>
#include "estimator/estimator.h"

class Manager : public ParamServer
{
    private:
        ros::Subscriber sub_imu;
        ros::Subscriber sub_cloud;
        ros::Subscriber sub_image;
        Estimator estimator;

    public:
        Manager()
        {
            sub_imu = nh.subscribe(imu_topic, 2000, &Manager::imuCallback, this, ros::TransportHints().tcpNoDelay());
            sub_image = nh.subscribe(img_topic, 100, &Manager::imgCallback, this, ros::TransportHints().tcpNoDelay());
            sub_cloud = nh.subscribe(cloud_topic, 5, &Manager::cloudCallback, this, ros::TransportHints().tcpNoDelay());
        }

        ~Manager(){}

        void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
        {
            double t = imu_msg->header.stamp.toSec();
            double dx = imu_msg->linear_acceleration.x;
            double dy = imu_msg->linear_acceleration.y;
            double dz = imu_msg->linear_acceleration.z;
            double rx = imu_msg->angular_velocity.x;
            double ry = imu_msg->angular_velocity.y;
            double rz = imu_msg->angular_velocity.z;
            Eigen::Vector3d acc(dx, dy, dz);
            Eigen::Vector3d gyr(rx, ry, rz);
            estimator.inputIMU(t, acc, gyr);
            return;
        }

        void imgCallback(const sensor_msgs::ImageConstPtr &img_msg)
        {

        }

        void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
        {
            double t = cloud_msg->header.stamp.toSec();
            estimator.inputCloud(t, cloud_msg);
            return;
        }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "fusion_estimator");

    Manager node;

    ROS_INFO("\033[1;32m----> Estimator node started.\033[0m");

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}