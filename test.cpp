#include "KMeans.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace cv;
using namespace std;

// Constructor
KMeans::KMeans(int clusters, int maxIterations, float epsilon)
    : k(clusters), maxIters(maxIterations), eps(epsilon) {}

// Euclidean distance between two 3D points (for RGB)
float KMeans::euclideanDistance(const Vec3f& a, const Vec3f& b) const {
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
}

// Initialize centroids randomly
void KMeans::initializeCentroids(const Mat& data) {
    srand(time(0)); // Seed random number generator
    centroids = Mat::zeros(k, 1, CV_32FC3);

    for (int i = 0; i < k; ++i) {
        int randomIndex = rand() % data.rows;
        centroids.at<Vec3f>(i) = data.at<Vec3f>(randomIndex);
    }
}

// Assign each data point to the nearest centroid
void KMeans::assignClusters(const Mat& data) {
    labels = Mat::zeros(data.rows, 1, CV_32S); // Labels for each data point

    for (int i = 0; i < data.rows; ++i) {
        float minDist = FLT_MAX;
        int bestCluster = 0;

        for (int j = 0; j < k; ++j) {
            float dist = euclideanDistance(data.at<Vec3f>(i), centroids.at<Vec3f>(j));
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }
        labels.at<int>(i) = bestCluster;
    }
}

// Update centroids based on cluster assignments
void KMeans::updateCentroids(const Mat& data) {
    vector<Vec3f> newCentroids(k, Vec3f(0, 0, 0));
    vector<int> clusterSizes(k, 0);

    for (int i = 0; i < data.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        newCentroids[clusterIdx] += data.at<Vec3f>(i);
        clusterSizes[clusterIdx]++;
    }

    for (int j = 0; j < k; ++j) {
        if (clusterSizes[j] > 0) {
            newCentroids[j] /= clusterSizes[j];
        }
        centroids.at<Vec3f>(j) = newCentroids[j];
    }
}

// Check if centroids have converged
bool KMeans::checkConvergence(const vector<Vec3f>& oldCentroids) const {
    for (int j = 0; j < k; ++j) {
        if (euclideanDistance(oldCentroids[j], centroids.at<Vec3f>(j)) > eps) {
            return false;
        }
    }
    return true;
}

// Fit K-Means to the data
void KMeans::fit(const Mat& data) {
    initializeCentroids(data);

    for (int iter = 0; iter < maxIters; ++iter) {
        // Store old centroids for convergence check
        vector<Vec3f> oldCentroids(k);
        for (int j = 0; j < k; ++j) {
            oldCentroids[j] = centroids.at<Vec3f>(j);
        }

        // Assign clusters and update centroids
        assignClusters(data);
        updateCentroids(data);

        // Check convergence
        if (checkConvergence(oldCentroids)) {
            cout << "Converged in " << iter + 1 << " iterations." << endl;
            break;
        }
    }
}

// Get cluster labels
Mat KMeans::getLabels() const {
    return labels;
}

// Get cluster centroids
Mat KMeans::getCentroids() const {
    return centroids;
}
