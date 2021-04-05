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
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(Ceres REQUIRED)

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

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(fusion_lib
	src/estimator/estimator.cpp
	)

target_link_libraries(fusion_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(fusion_node src/run.cpp)
target_link_libraries(fusion_node fusion_lib)
