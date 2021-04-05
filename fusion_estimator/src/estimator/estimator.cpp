#include "estimator.h"


Estimator::Estimator()
{
	ROS_INFO("Estimator begins");
	init_thread_flag = false;
	deskew_flag = false;
	ring_flag = false;
}

Estimator::~Estimator()
{

}

void Estimator::setParameter()
{
	if (!init_thread_flag)
	{
		init_thread_flag = true;
		process_thread = std::thread(&Estimator::processMeasurements, this);
	}
}

void Estimator::allocateMemory()
{
	laser_cloud_in.reset(new pcl::PointCloud<PointXYZIRT>());
	tmp_ouster_cloud_in.reset(new pcl::PointCloud<OusterPointXYZIRT>);
	full_cloud.reset(new pcl::PointCloud<PointType>());
	extracted_cloud.reset(new pcl::PointCloud<PointType>());

	full_cloud->points.resize(N_SCAN*HORIZON_SCAN);

	cloud_info.startRingIndex.assign(N_SCAN, 0);
	cloud_info.endRingIndex.assign(N_SCAN, 0);
	cloud_info.pointColInd.assign(N_SCAN*HORIZON_SCAN, 0);
	cloud_info.pointRange.assign(N_SCAN*HORIZON_SCAN, 0);
	// resetParameters();
}

bool Estimator::IMUAvailable(const double t1, const double t2)
{
	if (t2 == 0) // imu available for image
	{
		if (!acc_buf.empty() && t1 <= acc_buf.back().first)
			return true;
		else
			return false;
	}
	else // imu avaiable for pointcloud
	{
		if (!acc_buf.empty() && acc_buf.front().first < t1 && acc_buf.back().first > t2)
			return true;
		else
			return false;
	}

}

void Estimator::processMeasurements()
{
	while(1)
	{
		if (!cloud_buf.empty())
		{
			// Cache point cloud
			// - current_cloud
			// - scan_start_time
			// - scan_end_time
			if(!cachePointCloud())
			{
				printf("Waiting for valid pointcloud\n");
				return; 
			}

			// check imu
			while(1)
			{
				if (IMUAvailable(scan_start_time, scan_end_time))
					break;
				else
				{
					printf("waiting for imu...\n");
					std::chrono::milliseconds dura(5);
					std::this_thread::sleep_for(dura);
				}
			}
		}
	}
}

void Estimator::inputCloud(double t, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	m_buf.lock();
	cloud_buf.push(make_pair(t, *cloud_msg));
	m_buf.unlock();
}

bool Estimator::cachePointCloud()
{	
	if (cloud_buf.size() <= 2)
		return false;

	current_cloud = std::move(cloud_buf.front());
	cloud_buf.pop();

	// convert point cloud according to sensor type 
	if (sensor == SensorType::VELODYNE)
    {
        pcl::moveFromROSMsg(current_cloud.second, *laser_cloud_in);
    }
    else if (sensor == SensorType::OUSTER)
    {
        // Convert to Velodyne format
        pcl::moveFromROSMsg(current_cloud.second, *tmp_ouster_cloud_in);
        laser_cloud_in->points.resize(tmp_ouster_cloud_in->size());
        laser_cloud_in->is_dense = tmp_ouster_cloud_in->is_dense;
        for (size_t i = 0; i < tmp_ouster_cloud_in->size(); i++)
        {
            auto &src = tmp_ouster_cloud_in->points[i];
            auto &dst = laser_cloud_in->points[i];
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
    if (laser_cloud_in->is_dense == false)
    {
        ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
        ros::shutdown();
    }

    // get timestamp
    cloud_header = current_cloud.second.header;
    scan_start_time = current_cloud.first;
    scan_end_time = scan_start_time + laser_cloud_in->points.back().time;

    // check ring channel
    static int ring_flag = 0;
    if (ring_flag == 0)
    {
        ring_flag = -1;
        for (int i = 0; i < (int)current_cloud.second.fields.size(); ++i)
        {
            if (current_cloud.second.fields[i].name == "ring")
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
    if (deskew_flag == 0)
    {
        deskew_flag = -1;
        for (auto &field : current_cloud.second.fields)
        {
            if (field.name == "time" || field.name == "t")
            {
                deskew_flag = 1;
                break;
            }
        }
        if (deskew_flag == -1)
            ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
    }
    return true;
}

// void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
// {

// }

void Estimator::inputIMU(double t, const Vector3d &acc, const Vector3d &gyr)
{
	m_buf.lock();
	acc_buf.push(make_pair(t, acc));
	gyr_buf.push(make_pair(t, gyr));
	m_buf.unlock();

}

void Estimator::processIMU(double t, double dt, const Vector3d &acc, const Vector3d &gyr)
{
	
}