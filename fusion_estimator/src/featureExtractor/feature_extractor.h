#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "../utility/utility.h"
#include "../utility/parameters.h"
#include "fusion_estimator/CloudInfo.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtractor : public ParamServer
{

public:

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    fusion_estimator::CloudInfo cloudInfo;

    FeatureExtractor();

    ~FeatureExtractor();

    void initializationValue();

    fusion_estimator::CloudInfo extractEdgeSurfFeatures(fusion_estimator::CloudInfo &_cloudInfo);
    
    void calculateSmoothness();

    void markOccludedPoints();

    void extractFeatures();

    void freeCloudInfoMemory();

    void setCloudInfoEdgeSurf();
};

#endif