//
// Created by hoangqc on 18/03/2019.
//
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>

#include "Superpoint.h"

using namespace std;
using namespace cv;

cv::String modelPath = "../models/superpoint_v1_320x240.pt";
Superpoint engine;

int main()
{
//    cout << "Hello Superpoint" << endl;
    vector<cv::String> fn;
    cv::glob("../icl_snippet",fn,true);

    engine.init(modelPath, true, false);

    for( auto fname : fn)
    {
        Mat bgr_img = cv::imread(fname);
        engine.run(bgr_img);
    }

    cv::destroyAllWindows();
    return 0;
}