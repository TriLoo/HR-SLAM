#include "SIFT.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("lena.jpg", IMREAD_COLOR);

    imshow("Input", img);

    rectangle(img, Point(0, 0), Point(50, 50), Scalar(255, 255, 255), CV_FILLED);

    imshow("Output", img);

    waitKey(0);

    return 0;
}
