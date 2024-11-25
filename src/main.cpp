#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kmeans.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT); // Suppress logs generated by opencv

    String imagePath;
    int k;

    cout << "Enter image name [must be in same dir] (image.png): ";
    cin >> imagePath;

    String fileName = imagePath.substr(0, imagePath.find('.'));
    String ext = imagePath.substr(imagePath.find('.'), imagePath.length());

    cout << "\nEnter k value (number of colours): ";
    cin >> k;
    

    // Load Image
    Mat image = imread(imagePath);

    if (image.channels() == 1) {
        cvtColor(image, image, COLOR_GRAY2BGR);
    }

    if (image.empty()) {
        cout << "Could not load image!" << endl;
        return -1;
    }

    // Rescale Image for faster compression
    resize(image, image, Size(), 0.5, 0.5, INTER_AREA);

    Mat reshapedImage = image.reshape(3, image.rows * image.cols);
    reshapedImage.convertTo(reshapedImage, CV_32FC3); 

    KMeans KM(k);
    KM.fit(reshapedImage);

    Mat compressedImage = reshapedImage.clone();
    for (int i = 0; i < compressedImage.rows; ++i) {
        int clusterIdx = KM.getLabels().at<int>(i);
        compressedImage.at<Vec3f>(i) = KM.getCentroids().at<Vec3f>(clusterIdx);
    }

    compressedImage = compressedImage.reshape(3, image.rows);
    compressedImage.convertTo(compressedImage, CV_8U);

    cv::imwrite(fileName + "_compressed" + ext, compressedImage);

    cout << "Image saved as: " + fileName + ext << endl;

    return 0;
}
