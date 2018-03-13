//
// Created by smher on 18-3-8.
//

#include "myslam/Config.h"

namespace myFrontEnd
{
    // define the static variables
    std::shared_ptr<Config> Config::config_ = nullptr;
    // close files
    Config::~Config()
    {
        if (file_.isOpened())
            file_.release();
    }

    // set new parameter file
    void Config::setParameterFile(const std::string &filename)
    {
        if (config_ == nullptr)
            config_ = std::shared_ptr<Config>(new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false)
        {
            std::cerr << "Parameter file " << filename << " does not exist." << std::endl;
            config_->file_.release();
            return;
        }
    }


}

