//
// Created by smher on 18-3-16.
//

#include "myslam/PoseGraph.h"

namespace myBackEnd
{
    PoseGraph::PoseGraph()
    {}
    PoseGraph::~PoseGraph()
    {}

    PoseGraph& PoseGraph::createPG()
    {
        static PoseGraph pg;
        return pg;
    }
}
