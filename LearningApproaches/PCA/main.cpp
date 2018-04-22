#include "PCA.h"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // read data from file
    ifstream fin("test_data", fstream::in);
    assert(fin.is_open());

    Mat imgIn = imread("lena.jpg", IMREAD_COLOR);
    assert(!imgIn.empty());

    cout << "Image info: " << imgIn.rows << " * " << imgIn.cols << endl;

    Mat imgOut = Mat::zeros(imgIn.size(), CV_32FC1);
    LA::myPCA pca(5);                  // the first 5 components
    pca.calcPCA(imgOut, imgIn);

    //imgOut.convertTo(imgOut, CV_32FC1, 1.0/255);
    normalize(imgOut, imgOut, 0, 1.0, NORM_MINMAX);
    cout << "Size = " << imgOut.rows << " * " << imgOut.cols << endl;         // for test

    imshow("Results", imgOut);           // the Result is a 1 * 396 vector
    waitKey(0);

    // cout << IMREAD_GRAYSCALE << endl;            // cout 0

    return 0;
}
