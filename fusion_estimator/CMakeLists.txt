cmake_minimum_required(VERSION 3.0.2)
project(fusion_estimator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf
  cv_bridge
  image_transport
  pcl_conversions
  camera_models
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)

add_message_files(
  DIRECTORY msg
  FILES
  CloudInfo.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

catkin_package(
  DEPENDS PCL GTSAM
)

add_library(fusion_lib
	src/estimator/estimator.cpp
  src/featureExtractor/feature_extractor.cpp
  src/featureTracker/feature_tracker.cpp
  src/preintegration/preintegration.cpp
  src/mapOptimizer/map_optimizer.cpp
)

target_link_libraries(fusion_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} gtsam)

add_executable(fusion_node src/run.cpp)
target_link_libraries(fusion_node fusion_lib)
