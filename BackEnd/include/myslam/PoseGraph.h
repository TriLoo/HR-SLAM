//
// Created by smher on 18-3-16.
//

#ifndef BACKEND_POSEGRAPH_H
#define BACKEND_POSEGRAPH_H

#include "system.h"

namespace myBackEnd
{
    class PoseGraph
    {
    public:
        typedef std::shared_ptr<PoseGraph> Ptr;

        ~PoseGraph();

        // 单例模式
        PoseGraph& createPG();

    private:
        PoseGraph();   // 单例模式
    };
}

#endif //BACKEND_POSEGRAPH_H
