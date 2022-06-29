//
// Created by hoangqc on 27/06/2022.
//

#ifndef OV_CORE_GRAPHUTILS_H
#define OV_CORE_GRAPHUTILS_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct GraphImgInfo {
    // image sizes
    uint w, h, w_c, h_c;
    uint cell_size = 8;

    cv::Mat src_gray;
    cv::Mat src_viz;
    cv::Mat mask_grid;
    cv::Mat mask_edges;
    cv::Mat mask_points;

    cv::Mat debug_preview;


    //Processing struct
    cv::Mat graph_mask2D; //store masks for node & edges
    cv::Mat grad;

};

void drawGrids(cv::Mat & img, int step=8, cv::Scalar color= CV_RGB(255,0,0))
{
    uint w = img.size().width, h = img.size().height;
    uint w_steps = uint(w/step), h_steps = uint(h/step);

    for(uint i = 0; i < h_steps; i++)
        cv::line(img, cv::Point(0,i*step), cv::Point(w,i*step), color); // horizontal
    for(uint j = 0; j < w_steps; j++)
        cv::line(img, cv::Point(j*step, 0), cv::Point(j*step,h), color); // vertical
}

void calculate_gradient(const cv::Mat & src_gray, cv::Mat & out)
{
    Mat grad, src_blur;
    int ksize = 3, scale = 1, delta = 0, ddepth = CV_16S;

    GaussianBlur(src_gray, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_blur, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_blur, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
}

void genGraphInfo(GraphImgInfo & g, const cv::Mat & input)
{
    g.w = input.cols;
    g.h = input.rows;
    g.w_c = g.w/8;
    g.h_c = g.h/8;

    g.mask_grid = cv::Mat::zeros(input.size(), CV_8UC3);
    g.mask_edges = cv::Mat::zeros(input.size(), CV_8UC3);
    g.mask_points = cv::Mat::zeros(input.size(), CV_8UC3);

    g.graph_mask2D = cv::Mat::zeros(input.size()/8, CV_16S);

    drawGrids(g.mask_grid);

    cv::cvtColor(input, g.src_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(g.src_gray, g.src_viz, cv::COLOR_GRAY2BGR);
    cv::addWeighted(g.src_viz, 1.0, g.mask_grid, 0.3, 0, g.debug_preview);

    calculate_gradient(g.src_gray, g.grad);
}







#endif //OV_CORE_GRAPHUTILS_H
