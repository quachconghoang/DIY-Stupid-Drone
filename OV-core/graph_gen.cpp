#include <opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h>
//#include <cudnn.h>

//#include <iostream>
#include "ELSED/ELSED.h"
#include "DNN/Superpoint.h"
#include "DNN/anms.h"

using namespace cv;

void getGradientMask(cv::Mat & image)
{

}

int main()
{
    Mat image,src, src_gray;
    Mat grad;
    int ksize = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    image = cv::imread("../Data/viode_3.png");
    GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return 0;
}