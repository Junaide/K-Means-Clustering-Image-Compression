#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kmeans.hpp"

using namespace cv;
using namespace std;

int main() {
    // Step 1: Load the image
    Mat image = imread("Screenshot.jpg"); // Update the path to your image
    if (image.empty()) {
        cout << "Could not load image!" << endl;
        return -1;
    }

    // Step 2: Reshape the image to a 2D array of pixels
    Mat reshapedImage = image.reshape(1, image.rows * image.cols); // Flatten to (number_of_pixels, 3)
    reshapedImage.convertTo(reshapedImage, CV_32F); // Convert to float for K-Means

    int k = 16;

    KMeans KM(k);

    KM.fit(reshapedImage);
   

    return 0;
}
