#include <iostream>
#include "myslam/Config.h"

using namespace std;
/*
class testSingle
{
private:
    testSingle(){}

//   ~testSingle()
//   {
//       cout << "Private: " << endl;
//       cout << "In Deconstructor." << endl;
//   }
public:
    ~testSingle()
    {
        cout << "Public: " << endl;
        cout << "In Deconstructor." << endl;
    }

    static void Destroy()
    {
        //~testSingle();
    }

    static testSingle& Instance()
    {
        static testSingle theTestSingle;
        return theTestSingle;
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    cout << "Enter Local Area." << endl;
    {
        testSingle testS = testSingle::Instance();          // Error: the
        testS.Destroy();
    }
    cout << "Eixt Local Area." << endl;

    return 0;
}
*/

int main(int argc, char **argv)
{
    cout << "hello world." << endl;
    if (argc != 2)
    {
        cerr << "Usage: mySLAM configureFile.yaml" << endl;
        exit(EXIT_FAILURE);
    }

    string filename(argv[1]);
    cout << "Configure File Name: " << endl;

    myFrontEnd::Config::setParameterFile(filename);
    cout << "fx of camera = " << myFrontEnd::Config::getParam<double>("camera.fx") << endl;

    cout << "Done." << endl;

    return 0;
}
