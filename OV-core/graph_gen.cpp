#include <opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h>
//#include <cudnn.h>

//#include <iostream>
#include "ELSED/ELSED.h"
#include "DNN/Superpoint.h"
#include "DNN/GraphUtils.h"
#include "DNN/ANMS/anms.h"


using namespace cv;
using namespace std;


inline void drawSegments(cv::Mat img, upm::Segments segs, const cv::Scalar &color,
             int thickness = 1, int lineType = cv::LINE_AA, int shift = 0) {
//    int size = segs.size();
//    if(size > 100) size = 100;
//    for (uint i=0; i < size;i++)
//        cv::line(img, cv::Point2f(segs[i][0], segs[i][1]), cv::Point2f(segs[i][2], segs[i][3]),
//                 color, thickness, lineType, shift);
    for (const upm::Segment &seg: segs)
    {
        cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
        cv::drawMarker(img, cv::Point2f(seg[0], seg[1]), CV_RGB(255,0,0), MARKER_CROSS, 8);
        cv::drawMarker(img, cv::Point2f(seg[2], seg[3]), CV_RGB(255,0,0), MARKER_CROSS, 8);
    }

}

inline void drawSalients(cv::Mat img, upm::SalientSegments segs, const cv::Scalar &color,
                         int thickness = 1, int lineType = cv::LINE_AA, int shift = 0) {
    for (const upm::SalientSegment &seg: segs)
        cv::line(img, cv::Point2f(seg.segment[0], seg.segment[1]), cv::Point2f(seg.segment[2], seg.segment[3]),
                 color, thickness, lineType, shift);
}

string windows_src = "SRC";
string windows_prev = "SOURCE";
string windows_prev_target = "TARGET";

GraphImgInfo * g_glob[2];

static void on_trackbar( int, void* )
{

}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    GraphImgInfo g1 = *g_glob[0];//*(GraphImgInfo *)userdata;
    GraphImgInfo g2 = *g_glob[1];
    if  ( event == EVENT_LBUTTONDOWN )
    {
        genInteraction(g1,g2, x,y);
        imshow(windows_prev, g1.debug_preview);
        imshow(windows_prev_target, g2.debug_preview);
//        imshow(windows_src, info.grad);
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
//        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        genInteraction(g1,g2);
        imshow(windows_prev, g1.debug_preview);
    }
}


int test_graph()
{
    GraphImgInfo g1,g2;
    g_glob[0] = &g1;
    g_glob[1] = &g2;

    Mat image = cv::imread("../Data/frame000300.png");
    FileStorage fs("../Data/Keypoints000300.yml", FileStorage::READ);
    read(fs["keypoints"], g1.kpts);
    read(fs["descriptors"], g1.desc);

    Mat image2 = cv::imread("../Data/frame000310.png");
    FileStorage fs2("../Data/Keypoints000310.yml", FileStorage::READ);
    read(fs2["keypoints"], g2.kpts);
    read(fs2["descriptors"], g2.desc);

    fs.release();

    genGraphInfo(g1, image);
    genGraphInfo(g2, image2);
    genGraphCorrelation(g1,g2);
//    Mat A,B;
//    A = g1.desc;
//    cv::transpose(g2.desc, B);
//    Mat wtf = (A*B)<0.7;
//    Mat des0 = g1.desc.row(250);
//    for(int i=0; i<g1.kpts.size(); i++) {
//        Mat des1 = g1.desc.row(i);
//        double rs = cv::norm(des1,des0,NORM_L2);
//        if(rs < 1.0)
//        {
//            cv::drawMarker(g1.src_viz, g1.kpts[i].pt, CV_RGB(0, 255, 0),
//                           MARKER_CROSS, 9,2);
//        }
//    }
//    cv::drawMarker(g1.src_viz, g1.kpts[200].pt, CV_RGB(255, 0, 0), MARKER_CROSS, 9,2);


    namedWindow(windows_prev, 1);

//    createTrackbar( "My TrackBar", windows_prev, 0, 10, on_trackbar );
    setMouseCallback(windows_prev, CallBackFunc, &g1);
//    imshow("GRAD", g1.grad);
    imshow(windows_prev, g1.debug_preview);
    imshow(windows_prev_target, g2.debug_preview);

    int k = waitKey();

    return 0;
}

int test_graph_visualization()
{
    std::vector<cv::String> img_paths, kps_paths;
    cv::glob("../Data/frames/",img_paths,true);
    cv::glob("../Data/tmp/",kps_paths,true);

    for (int index = 0; index < 90; ++index)
    {
        Mat image = cv::imread(img_paths[index]);

        std::vector<cv::KeyPoint> kpts;
        cv::Mat desc;
        FileStorage fs(kps_paths[index], FileStorage::READ);
        read(fs["keypoints"], kpts);
        read(fs["descriptors"], desc);
        fs.release();

        GraphImgInfo g;
        genGraphInfo(g,image);
        for(int i=0;i<kpts.size();i++)
        {
            cv::drawMarker(g.debug_preview, kpts[i].pt, CV_RGB(255,255,0), MARKER_CROSS, 5);
//            std::cout << i << ": " << kpts[i].response << std::endl;
        }
//    drawKeypoints(g.debug_preview,kpts,g.debug_preview,CV_RGB(255,255,0));

        namedWindow("PREVIEWS", 1);
        setMouseCallback("PREVIEWS", CallBackFunc, &g);
        imshow("PREVIEWS", g.debug_preview);

        int k = waitKey();
        if (k==27) break;
    }

    return 0;
}

int test_superpoint()
{
    std::string home_dir = std::string(std::getenv("HOME"));
    std::string modelPath =  home_dir + "/Datasets/Weights/superpoint_v1_752x480.pt";
    Superpoint engine;
    engine.init(modelPath, false,true);
    torch::NoGradGuard no_grad;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

    cv::Mat image, src, src_viz;
    image = cv::imread("../Data/viode_1.png");
    cvtColor(image, src, COLOR_BGR2GRAY);
    cvtColor(src, src_viz, COLOR_GRAY2BGR);

    engine.compute_NN(image);
    engine.getKeyPoints(kpts,desc);

    drawKeypoints(src_viz,kpts,src_viz,CV_RGB(255,255,0));
    imshow(windows_src, src_viz);
    waitKey();

//    FileStorage fs("../Data/tmp/Keypoints.yml", FileStorage::WRITE);
//    write(fs, "keypoints", kpts);
//    write(fs, "descriptors", desc);
//    fs.release();

    return 0;
}

int test_superpoint_sequence()
{
    std::string home_dir = std::string(std::getenv("HOME"));
    std::string modelPath =  home_dir + "/Datasets/Weights/superpoint_v1_752x480.pt";
    Superpoint engine;
    engine.init(modelPath, false,true);
    torch::NoGradGuard no_grad;

    std::vector<cv::String> paths;
    cv::glob("../Data/frames/",paths,true);

    for(int i=0; i < paths.size(); i++)
    {
        std::vector<cv::KeyPoint> kpts;
        cv::Mat desc;
        cv::Mat image, src, src_viz;

        image = imread(paths[i]);
        cvtColor(image, src, COLOR_BGR2GRAY);
        cvtColor(src, src_viz, COLOR_GRAY2BGR);

        engine.compute_NN(image);
        engine.getKeyPoints(kpts,desc);


        int index = 300+i;
        std::string s = std::to_string(index);
        s.insert(0, 6 - s.length(), '0');
        std::string savePath = "../Data/tmp/Keypoints" + s + ".yml";
        cout << savePath << endl;
        FileStorage fs(savePath, FileStorage::WRITE);
        write(fs, "keypoints", kpts);
        write(fs, "descriptors", desc);
        fs.release();

        drawKeypoints(src_viz,kpts,src_viz,CV_RGB(255,255,0));
        imshow(windows_src, src_viz);
        waitKey(10);
    }
    return 0;
}

int test_edges()
{
    Mat image, src, src_blur, src_viz;
    Mat grad;
    int ksize = 3, scale = 1, delta = 0, ddepth = CV_16S;

    image = cv::imread("../Data/viode_1.png");

//    std::vector<cv::String> paths;
//    cv::glob("../Data/frames/",paths,true);


    cvtColor(image, src, COLOR_BGR2GRAY);
    cvtColor(src, src_viz, COLOR_GRAY2BGR);

    GaussianBlur(src, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_blur, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_blur, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    upm::ELSEDParams params;
    params.minLineLen = 15; //15
    params.gradientThreshold = 30;
    upm::ELSED elsed(params);
//    elsed.detectSalient()
    upm::Segments segs = elsed.detect(src);
    std::cout << "ELSED detected: " << segs.size() << " (large) segments" << std::endl;
    drawSegments(src_viz, segs, CV_RGB(0, 255, 0), 1);

//    cv::drawKeypoints(src_viz,kpts,src_viz,CV_RGB(255,255,0));

//    upm::SalientSegments salients = elsed.detectSalient(src);
//    drawSalients(src_viz, salients, CV_RGB(255,0,0), 1);
    cv::imshow("SRC", src);
    cv::imshow("ELSED long", src_viz);
    cv::imshow("GRAD", grad);
    cv::waitKey();

//    upm::ELSEDParams params;
//    params.listJunctionSizes = {};
//    upm::ELSED elsed_short(params);
//    upm::Segments segs = elsed_short.detect(src);
//    std::cout << "ELSED detected: " << segs.size() << " (short) segments" << std::endl;
//    drawSegments(src_viz, segs, CV_RGB(0, 255, 0), 2);
//    cv::imshow("ELSED short", src_viz);
//    cv::waitKey();

    return 0;
}

int main()
{
    test_graph();
//    test_superpoint();
//    test_superpoint_sequence();
//    test_graph_visualization();
    return 0;
}