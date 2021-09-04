#include <iostream>
//#include <gtsam/precompiled_header.h>

#include <gtsam/config.h>
#include <gtsam/global_includes.h>

#include <gtsam/geometry/Rot2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

using namespace std;
using namespace gtsam;
const double degree = M_PI / 180;

int main()
{
    std::cout<< "HELLO - GTSAM";
    //gtsam::assert_equal
    /* 
    
    */
    Rot2 prior = Rot2::fromAngle(30 * degree);
    prior.print("goal angle: ");
    auto model = noiseModel::Isotropic::Sigma(1, 1 * degree);
    Symbol key('x', 1);

    gtsam::NonlinearFactorGraph graph;
    graph.addPrior(key, prior, model);
    graph.print("full graph");

    Values initial;
    initial.insert(key, Rot2::fromAngle(20 * degree));
    initial.print("initial estimate: ");

    Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
    result.print("final result: ");

    return 0;
}