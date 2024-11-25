#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class KMeans {
public:
    KMeans(int clusters, int maxIterations = 50, float epsilon = 1e-1);

    void fit(const cv::Mat& data);

    
    cv::Mat getLabels() const;
    cv::Mat getCentroids() const;

private:
    // Number of clusters
    int k;

    // Maximum number of iterations
    int maxIters;

    // Convergence threshold
    float eps;

    // Cluster assignments for each data point
    cv::Mat labels;

    // Centroids for each cluster
    cv::Mat centroids;

    float euclideanDistance(const cv::Vec3f& a, const cv::Vec3f& b) const;
    void initializeCentroids(const cv::Mat& data);
    void assignClusters(const cv::Mat& data);
    void updateCentroids(const cv::Mat& data);
    bool checkConvergence(const std::vector<cv::Vec3f>& oldCentroids) const;
};

#endif // KMEANS_HPP