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

cv::String modelPath = "../../models/superpoint_v1_376x240.pt";
Superpoint engine;

int main()
{
    engine.init(modelPath, true, true);
    torch::NoGradGuard no_grad;

    cout << "Hello Superpoint" << endl;
    vector<cv::String> fL, fR;
    cv::glob("/home/hoangqc/Datasets/VIODE/city_day_3_high/left",fL,true);
    cv::glob("/home/hoangqc/Datasets/VIODE/city_day_3_high/right",fR,true);

    for( int it = 0; it < fL.size()-1; it++)
    {
        Mat bgr_L = cv::imread(fL[it]);
        Mat bgr_R = cv::imread(fL[it+1]);
        vector<KeyPoint> kpts1, kpts2;
        Mat desc1, desc2;
        engine.compute_NN(bgr_L);
        engine.getKeyPoints(kpts1,desc1);

        engine.compute_NN(bgr_R);
        engine.getKeyPoints(kpts2,desc2);



        //-- Step 2: Matching descriptor vectors
        Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2, true);
        std::vector< DMatch > bf_matches;
        std::vector< DMatch > bf_matches_good;
        matcher->match( desc1, desc2, bf_matches );
//        cout << bf_matches.size();
        for (size_t i = 0; i < bf_matches.size(); i++)
        {
            if (bf_matches[i].distance > 0.7f)
            {
                DMatch m = bf_matches[i];
                cv::Point2f p1 = kpts1[m.queryIdx].pt;
                cv::Point2f p2 = kpts2[m.trainIdx].pt;
                double res = cv::norm(p2-p1);//Euclidian distance
                if(res < 32)
                {
//                    cv::circle(bgr_R, p2, 2, Scalar(255,0,0),1, FILLED);
                    cv::line(bgr_R,p1,p2,Scalar(0,255,0), 2);
                }

            }
//                bf_matches_good.push_back(bf_matches[i]);
        }

        //-- Draw matches
//        Mat img_matches;
//        drawMatches( bgr_L, kpts1, bgr_R, kpts2, bf_matches_good, img_matches );
        //-- Show detected matches
//        imshow("Matches", img_matches );




//        for (int i = 0; i < kpts1.size() ; i++) {
//            cv::circle(bgr_L, kpts1[i].pt, 2, Scalar(255,0,0),2, FILLED);
//        }
//
        for (int i = 0; i < kpts2.size() ; i++) {
            cv::circle(bgr_R, kpts2[i].pt, 2, Scalar(255,0,0),2, FILLED);
        }
//        cv::imshow("left", bgr_L);
        cv::imshow("right", bgr_R);
        if(cv::waitKey(0)==27) break;
    }

    cv::destroyAllWindows();
    return 0;
}