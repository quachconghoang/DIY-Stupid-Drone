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

    cv::Mat masks;
    cv::Mat mask_grid;
//    cv::Mat mask_edges;
    cv::Mat mask_points;
    cv::Mat mask_canvas;

    cv::Mat debug_preview;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

    //Processing struct
    cv::Mat graph_mask2D; //store masks for node & edges
    cv::Mat grad;
    cv::Mat grad_score;

    cv::Mat norm_self;
    cv::Mat norm_cross;
};

void genGraphInfo(GraphImgInfo & g, const cv::Mat & input);
void drawGrids(cv::Mat & img, int step=8, cv::Scalar color= CV_RGB(255,0,0));
void calculate_gradient(const cv::Mat & src_gray, cv::Mat & out);


void drawGrids(cv::Mat & img, int step, cv::Scalar color)
{
    uint w = img.size().width, h = img.size().height;
    uint w_steps = uint(w/step), h_steps = uint(h/step);

    for(uint i = 0; i < h_steps; i++)
        cv::line(img, cv::Point(0,i*step), cv::Point(w,i*step), color); // horizontal
    for(uint j = 0; j < w_steps; j++)
        cv::line(img, cv::Point(j*step, 0), cv::Point(j*step,h), color); // vertical
}

void calculate_gradient(const cv::Mat & src_gray, cv::Mat & grad, cv::Mat & grad_score)
{
    Mat src_blur;
    int ksize = 3, scale = 1, delta = 0, ddepth = CV_16S;

    GaussianBlur(src_gray, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_blur, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_blur, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);

    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

//    grad = grad - 30;
//    cv::Size score_size = grad.size()/8;
//    grad_score = cv::Mat(score_size, CV_16UC1);
//
//    for(int i=0; i<score_size.width; i++){
//        for(int j=0; j<score_size.height; j++){
//            cv::Rect local(i*8,j*8,8,8);
//            cv::Scalar sum = cv::sum(grad(local));
//            grad_score.at<ushort>(j,i) = ushort(sum[0]);
//        }
//    }
}

void genGraphInfo(GraphImgInfo & g, const cv::Mat & input)
{
    g.w = input.cols;
    g.h = input.rows;
    g.w_c = g.w/8;
    g.h_c = g.h/8;

    g.mask_grid = cv::Mat::zeros(input.size(), CV_8UC3);
//    g.mask_edges = cv::Mat::zeros(input.size(), CV_8UC3);
    g.mask_points = cv::Mat::zeros(input.size(), CV_8UC3);
    g.mask_canvas = cv::Mat::zeros(input.size(), CV_8UC3);

    g.graph_mask2D = cv::Mat(input.size()/8, CV_16S);
    g.graph_mask2D.setTo(-1);

    cv::cvtColor(input, g.src_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(g.src_gray, g.src_viz, cv::COLOR_GRAY2BGR);

//    drawGrids(g.mask_grid);
    calculate_gradient(g.src_gray, g.grad, g.grad_score);
    g.graph_mask2D = cv::Mat(input.size()/8, CV_16S);
    g.graph_mask2D.setTo(-1);

    for(int i=0; i<g.kpts.size(); i++)
    {
        cv::drawMarker(g.mask_points, g.kpts[i].pt, CV_RGB(255, 255, 0), MARKER_CROSS, 5);
        cv::Point2i p(int(g.kpts[i].pt.x/8), int(g.kpts[i].pt.y/8));
        g.graph_mask2D.at<short>(p) = i;
    }

    cv::addWeighted(g.mask_grid, 0.5,  g.mask_points, 1, 0,  g.masks);
    cv::addWeighted(g.masks, 1,  g.mask_canvas, 1, 0,  g.masks);
    cv::addWeighted(g.src_viz, 1, g.masks, 0.5, 0, g.debug_preview);
}

void genGraphCorrelation(GraphImgInfo & gSource, GraphImgInfo & gTarget)
{
    std::vector<KeyPoint> kpts1 = gSource.kpts;
    std::vector<KeyPoint> kpts2 = gTarget.kpts;
    Mat desc1 = gSource.desc;
    Mat desc2 = gTarget.desc;

    int num_src_kpts = kpts1.size();
    int num_tar_kpts = kpts2.size();


    gSource.norm_self = cv::Mat::zeros(cv::Size(num_src_kpts,num_src_kpts), CV_64FC1);
    for(int i=0; i<num_src_kpts; i++){
        Mat d0 = desc1.row(i);
        for(int j=0; j<num_src_kpts; j++){
            Mat d1 = desc1.row(j);
            gSource.norm_self.at<double>(i,j) =cv::norm(d0,d1,NORM_L2);
        }
    }

    gTarget.norm_self = cv::Mat::zeros(cv::Size(num_tar_kpts,num_tar_kpts), CV_64FC1);
    for(int i=0; i<num_tar_kpts; i++){
        Mat d0 = desc2.row(i);
        for(int j=0; j<num_tar_kpts; j++){
            Mat d1 = desc2.row(j);
            gTarget.norm_self.at<double>(i,j) = cv::norm(d0,d1,NORM_L2);
        }
    }


    gSource.norm_cross = cv::Mat::zeros(cv::Size(num_tar_kpts,num_src_kpts), CV_64FC1);
    gTarget.norm_cross = cv::Mat::zeros(cv::Size(num_src_kpts,num_tar_kpts), CV_64FC1);
    for(int i=0; i<num_src_kpts; i++){
        Mat d0 = desc1.row(i);
        for(int j=0; j<num_tar_kpts; j++){
            Mat d1 = desc2.row(j);
            double rs = cv::norm(d0,d1,NORM_L2);
            gSource.norm_cross.at<double>(i,j) = rs;
            gTarget.norm_cross.at<double>(j,i) = rs;
        }
    }

//    Mat dif = gSource.norm_cross.t() - gTarget.norm_cross;
//    std::cout<<"non-zeros = " << cv::countNonZero(dif) << std::endl;
    //    cv::imshow("debug", dif);
//    waitKey();
}

void updateSuperMask(GraphImgInfo & g)
{

}

void genInteraction(GraphImgInfo & g1, GraphImgInfo & g2, int x=-1, int y=-1)
{
    if (x==-1 && y==-1)
    {
        g1.mask_canvas.setTo(0);
        cv::addWeighted(g1.mask_grid, 0.2, g1.mask_points, 0.5, 0, g1.masks);
        cv::addWeighted(g1.masks, 1, g1.mask_canvas, 1, 0, g1.masks);
        cv::addWeighted(g1.src_viz, 1, g1.masks, 1, 0, g1.debug_preview);
        return;
    }

    cv::Point p(x/8,y/8);
    short index = g1.graph_mask2D.at<short>(p);
    if (index == -1) {return;}

    cv::KeyPoint kp = g1.kpts[index];
    std::cout << kp.pt << " - index: " << index << " - rensponse: " << kp.response << std::endl;

    /* SELF Match */
    double radious = 50; //F-O-V matter;
    int num_target = g1.kpts.size();
    std::vector<int> pos_indices;
    std::vector<int> neg_indices;
    Mat self_match = g1.norm_self.row(index);
    //    double min,max;
//    cv::Point2i minLoc,maxLoc;
//    minMaxLoc( cross_match, &min, &max, &minLoc, &maxLoc);
    for(int i=0;i<num_target;i++){
        double dis = cv::norm(kp.pt-g1.kpts[i].pt);
        if(dis > 0 && dis < radious){
            double match_val = self_match.at<double>(0,i);
            if(match_val>1.5){neg_indices.push_back(i);}
            if(match_val<1.5){pos_indices.push_back(i);}
        }
    }

    g1.mask_canvas.setTo(0);
    cv::drawMarker(g1.mask_canvas, kp.pt, CV_RGB(0, 255, 0),
                   MARKER_CROSS, 11, 2);
    cv::circle(g1.mask_canvas, kp.pt, radious, CV_RGB(255, 255, 0));
    for(int i=0; i<pos_indices.size();i++){
        int match_id = pos_indices[i];
        KeyPoint kp = g1.kpts[match_id];
        int thickness = 1;
        if(kp.response > 0.3) thickness = 2;
        cv::drawMarker(g1.mask_canvas, kp.pt, CV_RGB(255, 255, 0),
                       MARKER_CROSS, 9, thickness);
    }

    for(int i=0; i<neg_indices.size();i++){
        int match_id = neg_indices[i];
        KeyPoint kp = g1.kpts[match_id];
        int thickness = 1;
        if(kp.response > 0.3) thickness = 2;
        cv::drawMarker(g1.mask_canvas, kp.pt, CV_RGB(255, 0, 0),
                       MARKER_CROSS, 9, thickness);
    }

    cv::addWeighted(g1.mask_grid, 0.2, g1.mask_points, 0.3, 0, g1.masks);
    cv::addWeighted(g1.masks, 1, g1.mask_canvas, 1, 0, g1.masks);
    cv::addWeighted(g1.src_viz, 1, g1.masks, 1, 0, g1.debug_preview);


/*CROSS MATCH*/
    num_target = g2.kpts.size();
    Mat cross_match = g1.norm_cross.row(index);//Thresh
    double min,max;
    cv::Point2i minLoc,maxLoc;
    minMaxLoc( cross_match, &min, &max, &minLoc, &maxLoc);
    double match_eps = 1.05;
    double match_thres = min*match_eps;
    if (match_thres > 1.3) match_thres = 1.3;

    std::vector<int> match_indices;
    for(int i=0;i<num_target;i++){
        if(cross_match.at<double>(0,i) < match_thres){
            match_indices.push_back(i);
        }
    }
    std::cout << "Find: "<< match_indices.size() << " match candidates \n";


    g2.mask_canvas.setTo(0);
    for(int i=0; i<match_indices.size();i++){
        int match_id = match_indices[i];
        KeyPoint kp = g2.kpts[match_id];
        int thickness = 1;
        if(kp.response > 0.3) thickness = 2;
        cv::drawMarker(g2.mask_canvas, g2.kpts[match_id].pt, CV_RGB(0, 255, 0),
                       MARKER_CROSS, 11, thickness);
        cv::circle(g2.mask_canvas, g2.kpts[match_id].pt, radious, CV_RGB(255, 255, 0));
    }
    cv::addWeighted(g2.mask_grid, 0.2, g2.mask_points, 0.3, 0, g2.masks);
    cv::addWeighted(g2.masks, 1, g2.mask_canvas, 1, 0, g2.masks);
    cv::addWeighted(g2.src_viz, 1, g2.masks, 1, 0, g2.debug_preview);

}







#endif //OV_CORE_GRAPHUTILS_H
