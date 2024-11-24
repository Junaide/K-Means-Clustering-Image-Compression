#include "KMeans.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace cv;
using namespace std;

KMeans::KMeans(int clusters, int maxIterations, float epsilon) 
    : k(clusters), maxIters(maxIterations), eps(epsilon) {}

float KMeans::euclideanDistance(const Vec3f& a, const Vec3f& b) const {
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
}

void KMeans::initializeCentroids(const Mat& data) {
    srand(time(0)); // Seed random number generator
    centroids = Mat::zeros(k, 1, CV_32FC3);

    for (int i = 0; i < k; ++i) {
        int randomIndex = rand() % data.rows;
        centroids.at<Vec3f>(i) = data.at<Vec3f>(randomIndex);
    }
}

void KMeans::fit(const Mat& data) {
    initializeCentroids(data);
}