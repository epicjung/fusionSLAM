#include "utility.h"
#include "tic_toc.h"

#include <cstdio>
#include <iostream>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"


class VisualFeature : public ParamServer
{
    private:
        ros::Subscriber subImg;
        ros::Publisher pubUV;

    public:
        std::mutex imgLock;
        std::mutex processLock;
        std::thread processThread;

        int row, col;
        cv::Mat imTrack;
        cv::Mat mask;
        cv::Mat fisheye_mask;
        cv::Mat prev_img, cur_img;
        vector<cv::Point2f> n_pts;
        vector<cv::Point2f> predict_pts;
        vector<cv::Point2f> predict_pts_debug;
        vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
        vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
        vector<cv::Point2f> pts_velocity, right_pts_velocity;
        vector<int> ids, ids_right;
        vector<int> track_cnt;
        map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
        map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
        map<int, cv::Point2f> prevLeftPtsMap;
        deque<sensor_msgs::Image> imgQueue;
        vector<camodocal::CameraPtr> m_camera;
        double cur_time;
        double prev_time;
        bool stereo_cam;
        int n_id;
        bool hasPrediction;

        VisualFeature()
        {
            subImg        = nh.subscribe<sensor_msgs::Image>(imgTopic, 1000, &VisualFeature::imgHandler, this, ros::TransportHints().tcpNoDelay());
            pubUV         = nh.advertise<sensor_msgs::PointCloud>("tracked_feature", 1000);
            stereo_cam = 0;
            n_id = 0;
            hasPrediction = false;
        };

        ~VisualFeature()
        {
            processThread.join();
        }

        void processImage()
        {
            while(1)
            {
                // ROS_INFO("B -- imgQueue size: %d\n", imgQueue.size());

                if (!imgQueue.empty())
                {
                    sensor_msgs::Image rosImage = imgQueue.front();
                    imgLock.lock();
                    imgQueue.pop_front();
                    imgLock.unlock();
                    cv::Mat thisImage = getImageFromMsg(rosImage);

                    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
                    // 스트레오 처리 필요
                    
                    featureFrame = trackImage(ROS_TIME(&rosImage), thisImage);
                    
                    int numPoints = featureFrame.size();  

                    sensor_msgs::PointCloud featuresMsg;
                    featuresMsg.header.stamp = rosImage.header.stamp;
                    featuresMsg.header.frame_id = "image";
                    featuresMsg.points.resize(numPoints);
                    featuresMsg.channels.resize(2);
                    featuresMsg.channels[0].name = "feature_id";
                    featuresMsg.channels[0].values.resize(numPoints);
                    featuresMsg.channels[1].name = "associated";
                    featuresMsg.channels[1].values.resize(numPoints);
                    int cnt = 0;
                    for(auto it = featureFrame.begin(); it != featureFrame.end(); ++it)
                    {
                        int id = it->first;
                        Eigen::Matrix<double, 7, 1> xyz_uv_vel = it->second[0].second;
                        featuresMsg.points[cnt].x = float(xyz_uv_vel(3, 0));
                        featuresMsg.points[cnt].y = float(xyz_uv_vel(4, 0));
                        featuresMsg.channels[0].values[cnt] = float(id);
                        featuresMsg.channels[1].values[cnt] = 0.0;
                        cnt++;
                    }
                    pubUV.publish(featuresMsg);
                }
                else
                {
                    
                }
                std::chrono::milliseconds dura(2);
                std::this_thread::sleep_for(dura);
            }
        }

        void imgHandler(const sensor_msgs::Image::ConstPtr& imgMsg)
        {
            imgLock.lock();
            imgQueue.push_back(*imgMsg);
            imgLock.unlock();
            // printf("img height: %d\n", imgMsg->height);
            // printf("img_width: %d\n", imgMsg->width);
        }
   
        cv::Mat getImageFromMsg(const sensor_msgs::Image img_msg)
        {
            cv_bridge::CvImageConstPtr ptr;
            if (img_msg.encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = img_msg.header;
                img.height = img_msg.height;
                img.width = img_msg.width;
                img.is_bigendian = img_msg.is_bigendian;
                img.step = img_msg.step;
                img.data = img_msg.data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat img = ptr->image.clone();
            return img;
        }

        void setMask()
        {
            mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

            // prefer to keep features that are tracked for long time
            vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

            for (unsigned int i = 0; i < cur_pts.size(); i++)
                cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

            // track_cnt가 높은 순으로 정렬
            sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
                {
                    return a.first > b.first;
                });

            cur_pts.clear();
            ids.clear();
            track_cnt.clear();

            for (auto &it : cnt_pts_id)
            {
                if (mask.at<uchar>(it.second.first) == 255)
                {
                    cur_pts.push_back(it.second.first);
                    ids.push_back(it.second.second);
                    track_cnt.push_back(it.first);
                    cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
                }
            }
        }

        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat())
        {
            TicToc t_r;
            cur_time = _cur_time;
            cur_img = _img;
            row = cur_img.rows;
            col = cur_img.cols;
            cv::Mat rightImg = _img1;
            /*
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(cur_img, cur_img);
                if(!rightImg.empty())
                    clahe->apply(rightImg, rightImg);
            }
            */
            cur_pts.clear();

            // Feature points가 있을 때
            if (prev_pts.size() > 0)
            {
                TicToc t_o;
                vector<uchar> status;
                vector<float> err;
                if(hasPrediction) // setPrediction 함수 호출 시 hasPrediction = true
                {
                    cur_pts = predict_pts;
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                    
                    int succ_num = 0;
                    for (size_t i = 0; i < status.size(); i++)
                    {
                        if (status[i])
                            succ_num++;
                    }
                    if (succ_num < 10)
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
                }
                else
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
                
                /* 
                * calcOpticalFlowPyrLK() 
                * Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
                * 1st: prevImag
                * 2nd: 
                */

                // reverse check
                if(FLOW_BACK)
                {
                    vector<uchar> reverse_status;
                    vector<cv::Point2f> reverse_pts = prev_pts;
                    cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                    //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                        {
                            status[i] = 1;
                        }
                        else
                            status[i] = 0;
                    }
                }
                
                for (int i = 0; i < int(cur_pts.size()); i++)
                    if (status[i] && !inBorder(cur_pts[i]))
                        status[i] = 0;
                reduceVector(prev_pts, status);
                reduceVector(cur_pts, status);
                reduceVector(ids, status);
                reduceVector(track_cnt, status);
                ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
                //printf("track cnt %d\n", (int)ids.size());
            }


            for (auto &n : track_cnt)
                n++;

            if (1)
            {
                //rejectWithF();
                ROS_DEBUG("set mask begins");
                TicToc t_m;
                setMask(); // cur_pts가 clear() 되는데? n_max_cnt 무조건 MAX_CNT아닌가?
                // 예상: 이전 pts에서의 주변으로만 masking을 해서 feature를 찾기 위함?
                ROS_DEBUG("set mask costs %fms", t_m.toc());

                ROS_DEBUG("detect feature begins");
                TicToc t_t;
                int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size()); // 처음에는 cur_pts.size() = 0
                if (n_max_cnt > 0)
                {
                    if(mask.empty())
                        cout << "mask is empty " << endl;
                    if (mask.type() != CV_8UC1)
                        cout << "mask type wrong " << endl;
                    cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
                    /* 
                    * cv::goodFeaturesToTrack()
                    * Determines strong corners on an image
                    * 1st: input image
                    * 2nd: n_pts는 (u, v)좌표로 된 vector임 
                    * 3rd: max corner numbers
                    * 4th: quality level
                    * 5th: Minimum possible Euclidean distance b/t the returned corner 
                    * 6th: ROI (맨 처음 cur_pts가 없을 때는 mask는 empty; but ... later)
                    */
                }
                else
                    n_pts.clear();
                ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

                // 현재 이미지에서 찾은 feature points들 cur_pts, ids, track_cnt에 추가
                // * n_id는 초기화 되지 않고 계속 늘어 남 
                for (auto &p : n_pts)
                {
                    cur_pts.push_back(p);
                    ids.push_back(n_id++);
                    track_cnt.push_back(1);
                }
                // printf("feature cnt after add %d\n", (int)ids.size());
            }

            cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
            pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

            // cur_un_pts_map, prev_un_pts_map = [id, pt]

            if(!_img1.empty() && stereo_cam)
            {
                ids_right.clear();
                cur_right_pts.clear();
                cur_un_right_pts.clear();
                right_pts_velocity.clear();
                cur_un_right_pts_map.clear();
                if(!cur_pts.empty())
                {
                    //printf("stereo image; track feature on right image\n");
                    vector<cv::Point2f> reverseLeftPts;
                    vector<uchar> status, statusRightLeft;
                    vector<float> err;
                    // cur left ---- cur right
                    cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
                    // reverse check cur right ---- cur left
                    if(FLOW_BACK)
                    {
                        cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                        for(size_t i = 0; i < status.size(); i++)
                        {
                            if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                                status[i] = 1;
                            else
                                status[i] = 0;
                        }
                    }

                    ids_right = ids;
                    reduceVector(cur_right_pts, status);
                    reduceVector(ids_right, status);
                    // only keep left-right pts
                    /*
                    reduceVector(cur_pts, status);
                    reduceVector(ids, status);
                    reduceVector(track_cnt, status);
                    reduceVector(cur_un_pts, status);
                    reduceVector(pts_velocity, status);
                    */
                    cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
                    right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
                }
                prev_un_right_pts_map = cur_un_right_pts_map;
            }

            if(SHOW_TRACK)
                drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

            prev_img = cur_img;
            prev_pts = cur_pts;
            prev_un_pts = cur_un_pts; //undistorted points
            prev_un_pts_map = cur_un_pts_map; // [id, pt] map
            prev_time = cur_time;
            hasPrediction = false;

            // STEREO
            prevLeftPtsMap.clear();
            for(size_t i = 0; i < cur_pts.size(); i++)
                prevLeftPtsMap[ids[i]] = cur_pts[i];

            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
            // map<feature_id, vector<pair<camera_id, (x,y,1,u,v,vel_x,vel_y)>>>
            // x,y는 undistorted point 좌표
            // u,v는 original point 좌표
            // MONO일 경우 camera_id = 0
            for (size_t i = 0; i < ids.size(); i++)
            {
                int feature_id = ids[i];
                double x, y ,z;
                x = cur_un_pts[i].x;
                y = cur_un_pts[i].y;
                z = 1;
                double p_u, p_v;
                p_u = cur_pts[i].x;
                p_v = cur_pts[i].y;
                int camera_id = 0;
                double velocity_x, velocity_y;
                velocity_x = pts_velocity[i].x;
                velocity_y = pts_velocity[i].y;

                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            if (!_img1.empty() && stereo_cam)
            {
                for (size_t i = 0; i < ids_right.size(); i++)
                {
                    int feature_id = ids_right[i];
                    double x, y ,z;
                    x = cur_un_right_pts[i].x;
                    y = cur_un_right_pts[i].y;
                    z = 1;
                    double p_u, p_v;
                    p_u = cur_right_pts[i].x;
                    p_v = cur_right_pts[i].y;
                    int camera_id = 1;
                    double velocity_x, velocity_y;
                    velocity_x = right_pts_velocity[i].x;
                    velocity_y = right_pts_velocity[i].y;

                    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                    featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
                }
            }

            printf("feature track whole time %f\n", t_r.toc());
            return featureFrame;
        }

        void readIntrinsicParameter(const vector<string> &calib_file)
        {
            for (size_t i = 0; i < calib_file.size(); i++)
            {
                ROS_DEBUG("reading paramerter of camera %s", calib_file[i].c_str());
                FILE *fh = fopen(calib_file[i].c_str(), "r");
                if (fh == NULL)
                {
                    ROS_WARN("config_file doesn't exist");
                    ROS_BREAK();
                    return;
                }
                fclose(fh);

                camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
                m_camera.push_back(camera);
            }
            if (calib_file.size() == 2)
                stereo_cam = 1;
        }

        void showUndistortion(const string &name)
        {
            cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
            vector<Eigen::Vector2d> distortedp, undistortedp;
            for (int i = 0; i < col; i++)
                for (int j = 0; j < row; j++)
                {
                    Eigen::Vector2d a(i, j);
                    Eigen::Vector3d b;
                    m_camera[0]->liftProjective(a, b);
                    distortedp.push_back(a);
                    undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
                    //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
                }
            for (int i = 0; i < int(undistortedp.size()); i++)
            {
                cv::Mat pp(3, 1, CV_32FC1);
                pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
                pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
                pp.at<float>(2, 0) = 1.0;
                //cout << trackerData[0].K << endl;
                //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
                //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
                if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
                {
                    undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
                }
                else
                {
                    //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
                }
            }
            // turn the following code on if you need
            // cv::imshow(name, undistortedImg);
            // cv::waitKey(0);
        }

        vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
        {
            vector<cv::Point2f> un_pts;
            for (unsigned int i = 0; i < pts.size(); i++)
            {
                Eigen::Vector2d a(pts[i].x, pts[i].y);
                Eigen::Vector3d b;
                cam->liftProjective(a, b);
                un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
            }
            return un_pts;
        }

        vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
        {
            vector<cv::Point2f> pts_velocity;
            cur_id_pts.clear();
            for (unsigned int i = 0; i < ids.size(); i++)
            {
                cur_id_pts.insert(make_pair(ids[i], pts[i]));
            }

            // caculate points velocity
            // 이전 image에서의 feature와 현재 image에서의 feature가 같은 id를 갖는다는 가정
            // ROI에 의해 그 주변에서만 찾아서 그런가? 
            if (!prev_id_pts.empty())
            {
                double dt = cur_time - prev_time;
                
                for (unsigned int i = 0; i < pts.size(); i++)
                {
                    std::map<int, cv::Point2f>::iterator it;
                    it = prev_id_pts.find(ids[i]);
                    if (it != prev_id_pts.end())
                    {
                        double v_x = (pts[i].x - it->second.x) / dt;
                        double v_y = (pts[i].y - it->second.y) / dt;
                        pts_velocity.push_back(cv::Point2f(v_x, v_y));
                    }
                    else
                        pts_velocity.push_back(cv::Point2f(0, 0));

                }
            }
            else
            {
                for (unsigned int i = 0; i < cur_pts.size(); i++)
                {
                    pts_velocity.push_back(cv::Point2f(0, 0));
                }
            }
            return pts_velocity;
        }        

        void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                    vector<int> &curLeftIds,
                                    vector<cv::Point2f> &curLeftPts, 
                                    vector<cv::Point2f> &curRightPts,
                                    map<int, cv::Point2f> &prevLeftPtsMap)
        {
            //int rows = imLeft.rows;
            int cols = imLeft.cols;
            if (!imRight.empty() && stereo_cam)
                cv::hconcat(imLeft, imRight, imTrack);
            else
                imTrack = imLeft.clone();
            cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

            for (size_t j = 0; j < curLeftPts.size(); j++)
            {
                double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
                cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }
            if (!imRight.empty() && stereo_cam)
            {
                for (size_t i = 0; i < curRightPts.size(); i++)
                {
                    cv::Point2f rightPt = curRightPts[i];
                    rightPt.x += cols;
                    cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
                    //cv::Point2f leftPt = curLeftPtsTrackRight[i];
                    //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
                }
            }
            
            map<int, cv::Point2f>::iterator mapIt;
            for (size_t i = 0; i < curLeftIds.size(); i++)
            {
                int id = curLeftIds[i];
                mapIt = prevLeftPtsMap.find(id);
                if(mapIt != prevLeftPtsMap.end())
                {
                    cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
                }
            }

            //draw prediction
            /*
            for(size_t i = 0; i < predict_pts_debug.size(); i++)
            {
                cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
            }
            */
            //printf("predict pts size %d \n", (int)predict_pts_debug.size());

            //cv::Mat imCur2Compress;
            //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
        }


        void setPrediction(map<int, Eigen::Vector3d> &predictPts)
        {
            hasPrediction = true;
            predict_pts.clear();
            predict_pts_debug.clear();
            map<int, Eigen::Vector3d>::iterator itPredict;
            for (size_t i = 0; i < ids.size(); i++)
            {
                //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
                int id = ids[i];
                itPredict = predictPts.find(id);
                if (itPredict != predictPts.end())
                {
                    Eigen::Vector2d tmp_uv;
                    m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
                    predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
                    predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
                }
                else
                    predict_pts.push_back(prev_pts[i]);
            }
        }

        void removeOutliers(set<int> &removePtsIds)
        {
            std::set<int>::iterator itSet;
            vector<uchar> status;
            for (size_t i = 0; i < ids.size(); i++)
            {
                itSet = removePtsIds.find(ids[i]);
                if(itSet != removePtsIds.end())
                    status.push_back(0);
                else
                    status.push_back(1);
            }

            reduceVector(prev_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
        }


        cv::Mat getTrackImage()
        {
            return imTrack;
        }

        bool inBorder(const cv::Point2f &pt)
        {
            const int BORDER_SIZE = 1;
            int img_x = cvRound(pt.x);
            int img_y = cvRound(pt.y);
            return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
        }

        double distance(cv::Point2f pt1, cv::Point2f pt2)
        {
            //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
            double dx = pt1.x - pt2.x;
            double dy = pt1.y - pt2.y;
            return sqrt(dx * dx + dy * dy);
        }

        void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
        {
            int j = 0;
            for (int i = 0; i < int(v.size()); i++)
                if (status[i])
                    v[j++] = v[i];
            v.resize(j);
        }

        void reduceVector(vector<int> &v, vector<uchar> status)
        {
            int j = 0;
            for (int i = 0; i < int(v.size()); i++)
                if (status[i])
                    v[j++] = v[i];
            v.resize(j);
        }

        void setParameter()
        {
            readIntrinsicParameter(CAM_NAMES);
            processLock.lock();
            processThread = std::thread(&VisualFeature::processImage, this);
            processLock.unlock();
        }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_fusion");
    
    VisualFeature tracker;

    tracker.setParameter();

    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    
    // ros::MultiThreadedSpinner spinner(4);
    // spinner.spin();
    
    ros::spin();

    return 0;
}