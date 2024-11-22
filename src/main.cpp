#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("Screenshot.png"); // Update the path to your image
    if (image.empty()) {
        cout << "Could not load image!" << endl;
        return -1;
    }

    imshow("Image", image);
    waitKey(0); // Wait for a key press
    return 0;
}