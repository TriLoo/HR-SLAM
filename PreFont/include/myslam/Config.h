//
// Created by smher on 18-3-8.
//

#ifndef SLAMPREFONT_CONFIG_H
#define SLAMPREFONT_CONFIG_H

#include "system.h"

namespace myFrontEnd
{
    class Config
    {
    private:
        //typedef std::shared_ptr<Config> Ptr;
        // 饿汉式
        static std::shared_ptr<Config> config_;             // Static: 单例模式
        Config(){}                    // private constructor, making this class singleton

        cv::FileStorage file_;        // wirte or read XML / YAML format files
    public:
        ~Config();

        // set a new config file
        static void setParameterFile(const std::string& filename);

        // get the parameter value
        // 注意：模板参数 _T 出现在了返回值的位置，因此使用时，需要显示指定其类型
        template <typename _T>
                static _T getParam(const std::string& key)
        {
            return _T(config_->file_[key]);
        }

        // test by myself singleton
        // best implementation of singleton
        // 懒汉式
        static Config& Instance()
        {
            static Config theConfig;
            return theConfig;
        }
    };
}

#endif //SLAMPREFONT_CONFIG_H
