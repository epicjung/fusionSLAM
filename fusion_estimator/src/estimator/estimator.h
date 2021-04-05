
#include <thread>
#include <mutex>
#include <queue>
#include "fusion_estimator/CloudInfo.h"
#include "../utility/utility.h"
using namespace std;
using namespace Eigen;


class Estimator : public ParamServer
{
	public:
		Estimator();
		~Estimator();
		void setParameter();
		void allocateMemory();
		// data processing
		void inputCloud(double t, const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
		bool cachePointCloud();
		void inputIMU(double t, const Vector3d &acc, const Vector3d &gyr);
		bool IMUAvailable(const double t1, const double t2 = 0.0);
		void processIMU(double t, double dt, const Vector3d &acc, const Vector3d &gyr);
		void processMeasurements();

		bool init_thread_flag;
		bool deskew_flag;
		bool ring_flag;
		
		mutex m_buf;
		mutex m_process;
		mutex m_propagate;
		queue<pair<double, Vector3d>> acc_buf;
		queue<pair<double, Vector3d>> gyr_buf;
		queue<pair<double, sensor_msgs::PointCloud2>> cloud_buf;
		thread process_thread;

		pair<double, sensor_msgs::PointCloud2> current_cloud;
		pcl::PointCloud<PointXYZIRT>::Ptr laser_cloud_in;
		pcl::PointCloud<OusterPointXYZIRT>::Ptr tmp_ouster_cloud_in;
		pcl::PointCloud<PointType>::Ptr full_cloud;
		pcl::PointCloud<PointType>::Ptr extracted_cloud;
		fusion_estimator::CloudInfo cloud_info;
		std_msgs::Header cloud_header;
		double scan_start_time;
		double scan_end_time;

};