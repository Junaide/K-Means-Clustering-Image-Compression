#include "KMeans.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

KMeans::KMeans(int clusters, int maxIterations, float epsilon)
    : k(clusters), maxIters(maxIterations), eps(epsilon) {}

cv::Mat KMeans::getCentroids() const {
    return centroids.clone();
}

cv::Mat KMeans::getLabels() const {
    return labels.clone();
}
float KMeans::euclideanDistance(const Vec3f& a, const Vec3f& b) const {
    return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2);
}

void KMeans::initializeCentroids(const Mat& data) {

    srand(time(0)); 
    centroids = Mat::zeros(k, 1, CV_32FC3);

    for (int i = 0; i < k; ++i) {
        int randomIndex = rand() % data.rows;
        centroids.at<Vec3f>(i) = data.at<Vec3f>(randomIndex);
    }
}

void KMeans::assignClusters(const Mat& data) {
    labels = Mat::zeros(data.rows, 1, CV_32S);

    const Vec3f* dataPtr = data.ptr<Vec3f>();
    int* labelsPtr = labels.ptr<int>();

    #pragma omp parallel for
    for (int i = 0; i < data.rows; ++i) {
        Vec3f pixel = dataPtr[i];
        float minDist = FLT_MAX;
        int bestCluster = 0;

        for (int j = 0; j < k; ++j) {
            float dist = (pixel - centroids.at<Vec3f>(j)).dot(pixel - centroids.at<Vec3f>(j));
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;

                if (minDist == 0.0f) break;
            }
        }
        labelsPtr[i] = bestCluster;
    }
}



void KMeans::updateCentroids(const Mat& data) {
    vector<vector<Vec3f>> localCentroids(omp_get_max_threads(), vector<Vec3f>(k, Vec3f(0, 0, 0)));
    vector<vector<int>> localClusterSizes(omp_get_max_threads(), vector<int>(k, 0));

    const Vec3f* dataPtr = data.ptr<Vec3f>();
    const int* labelsPtr = labels.ptr<int>();

    #pragma omp parallel for
    for (int i = 0; i < data.rows; ++i) {
        int threadId = omp_get_thread_num();
        int currCluster = labelsPtr[i];
        localCentroids[threadId][currCluster] += dataPtr[i];
        localClusterSizes[threadId][currCluster]++;
    }

    vector<Vec3f> newCentroids(k, Vec3f(0, 0, 0));
    vector<int> clusterSizes(k, 0);

    for (int t = 0; t < omp_get_max_threads(); ++t) {
        for (int j = 0; j < k; ++j) {
            newCentroids[j] += localCentroids[t][j];
            clusterSizes[j] += localClusterSizes[t][j];
        }
    }

    for (int j = 0; j < k; ++j) {
        if (clusterSizes[j] > 0) {
            newCentroids[j] /= clusterSizes[j];
        } else {
            newCentroids[j] = dataPtr[rand() % data.rows];
        }
        centroids.at<Vec3f>(j) = newCentroids[j];
    }
}


void KMeans::fit(const Mat& data) {
    initializeCentroids(data);

    for(int i = 0; i < maxIters; ++i) {
    Mat prevCentroids = centroids.clone();

        assignClusters(data);
        updateCentroids(data);


        float variance = 0;
        for(int j = 0; j < k; ++j) {
            variance = max(variance, euclideanDistance(prevCentroids.at<Vec3f>(j), centroids.at<Vec3f>(j)));
        }

        if (variance < eps) { // Convergence condition
        break;
        }
    }
}
