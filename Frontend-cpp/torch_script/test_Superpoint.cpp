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

cv::String modelPath = "../../models/superpoint_v1_320x240.pt";
Superpoint engine;

int main()
{
    engine.init(modelPath, true, true);
    torch::NoGradGuard no_grad;

    cout << "Hello Superpoint" << endl;
    vector<cv::String> fn;
    cv::glob("../../icl_snippet",fn,true);

    for( auto fname : fn)
    {
         cout << fname <<endl;
        Mat bgr_img = cv::imread(fname);
//        cv::imshow("img", bgr_img);

        engine.run(bgr_img);
        cv::waitKey(100);
    }

//    VideoCapture cap;
//    if(!cap.open(0)) return 0;
//    for(;;){
//        Mat bgr_img;
//        cap >> bgr_img;
//        if( bgr_img.empty() ) break; // end of video stream
//        engine.run(bgr_img);
//    }

    cv::destroyAllWindows();
    return 0;
}