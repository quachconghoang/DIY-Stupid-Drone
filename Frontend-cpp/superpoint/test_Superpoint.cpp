//
// Created by hoangqc on 18/03/2019.
//
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>

#include "SuperPoint.h"

using namespace std;
using namespace cv;

cv::String modelPath = "../../models/superpoint.pt";
//cv::String modelPath = "../../models/superpoint_v1_320x240.pt";
std::shared_ptr<SuperPoint> model;

int main()
{
//    engine.init(modelPath, true, true);
    model = make_shared<SuperPoint>();
    torch::load(model, modelPath);
    SPDetector detector(model);

    cout << "Hello Superpoint" << endl;
    vector<cv::String> fn;
    cv::glob("../../icl_snippet",fn,true);
    Mat im_gray, bgr_img;

    for( auto fname : fn)
    {
        cout << fname <<endl;

        bgr_img = cv::imread(fname);
        cv::cvtColor(bgr_img, im_gray, cv::COLOR_BGR2GRAY);
//        cv::resize(im_gray, im_gray, cv::Size(320,240));
        detector.detect(im_gray,false);
        cv::imshow("img", im_gray);
        cv::waitKey(100);
//        engine.run(bgr_img);
    }

    cv::destroyAllWindows();
    return 0;
}