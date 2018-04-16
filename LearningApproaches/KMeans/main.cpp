#include "KMeans.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    Mat img = imread("lena.jpg", IMREAD_COLOR);
    img.convertTo(img, CV_32FC3, 1.0/255);

    Mat imgOut = Mat::zeros(img.size(), CV_32FC1);

    imshow("Input", img);

    KMeans km(3);
    km.calcKMeans(imgOut, img);

    imshow("Output", imgOut);

    waitKey(0);

    return 0;
}
