//
// Created by hoangqc on 18/03/2019.
//

#ifndef TEST_LIBTORCH_SUPERPOINT_H
#define TEST_LIBTORCH_SUPERPOINT_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

class Superpoint_tracker{

};

class Superpoint {

public:
    int W = 320;
    int H = 240;
    int cell = 8;
    float thres = 0.015f;
    int dist_thresh = 4;
    int border_remove = 4;

    bool m_debug = false;
    bool m_use_cuda = true;

    Superpoint();
    ~Superpoint();

    void init(const cv::String & model_path, bool debug=false, bool use_cuda = true);
    void run(cv::Mat & bgr_img);
    void clear();


private:
    std::shared_ptr<torch::jit::script::Module> module;
    std::vector<torch::jit::IValue> inputs;
    c10::intrusive_ptr<torch::ivalue::Tuple> outputs;

    std::vector<cv::Point> m_pts_nms;
    at::Tensor m_desc;


};



#endif //TEST_LIBTORCH_SUPERPOINT_H
