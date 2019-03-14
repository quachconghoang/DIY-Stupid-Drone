#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>

using namespace std;
using namespace cv;

cv::String modelPath = "../models/superpoint_v1_320x240.pt";
cv::String img_path = "../icl_snippet/250.png";

torch::Tensor nms_fast( const torch::Tensor& in_corners, int H, int W, float dist_thresh)
{
    torch::Tensor grid = torch::zeros({H,W}, torch::kInt64);
    torch::Tensor inds = torch::zeros({H,W}, torch::kInt64);

    torch::Tensor rs;
    return rs;
}

torch::Tensor imgFile_to_tensor(cv::String file, int img_H, int img_W)
{
    cv::Mat im = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (im.empty()) return torch::Tensor(); //only proceed if sucsessful
    cv::resize(im, im, Size(img_W, img_H));
//    cv::imshow("test",im);
//    cv::waitKey();

    torch::Tensor tensor_image = torch::from_blob(im.data, {1, im.rows, im.cols, 1}, at::kByte);
    tensor_image = tensor_image.to(at::kFloat)/255.f;
    tensor_image = at::transpose(tensor_image, 1, 2);
    tensor_image = at::transpose(tensor_image, 1, 3);

    return  tensor_image;
}

int W = 320;
int H = 240;
int cell = 8;
float thres = 0.015f;

int main()
{
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);
    assert(module != nullptr);
    std::cout << "cpu ok \n";
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
//        std::cout << "CUDA is available!" << std::endl;
        device = torch::Device(torch::kCUDA);
        module->to(device);
        std::cout <<  "gpu ok \n";
    }

//    vector<cv::String> fn;
//    cv::glob(img_path,fn,true);
    at::Tensor inp = imgFile_to_tensor(img_path, H, W);


    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(inp.cuda());

    auto outputs = module->forward(inputs).toTuple();
    torch::Tensor semi = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();
    torch::Tensor coarse_desc = outputs->elements()[1].toTensor().to(at::kCPU).squeeze();

    torch::Tensor dense = semi.exp();
    dense = dense / (torch::sum(dense,0) + .00001);

    torch::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);
    cout << nodust.sizes() << endl;

    int Hc = int(H / cell);
    int Wc = int(W / cell);

    torch::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});

    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({Hc*cell, Wc*cell});
    cout << heatmap.sizes() << endl;

    torch::Tensor tmp_loc = (heatmap >= thres).nonzero();//.transpose(0,1);
    cout << tmp_loc.sizes() << endl;
    cout << tmp_loc;
//    cout << heatmap << endl;
//    cv::Mat tmp = cv::Mat(H,W, CV_32F, heatmap.data_ptr());
//    cv::imshow("test", tmp); cv::waitKey(); cv::destroyAllWindows();
}

int mainX()
{
    torch::Tensor semi = torch::arange(0, 80).reshape({5,4,4})/10;
    torch::Tensor dense = semi.exp();
    dense = dense / (torch::sum(dense,0) + .00001);

    torch::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);
    cout << nodust.sizes() << endl;

    torch::Tensor heatmap = nodust.reshape({4,4,2,2});
    cout << heatmap.sizes() << endl;
    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({8,8});
    cout << heatmap << endl;

    // WARNING --- ROW-COL to XYZ
    torch::Tensor tmp_loc = (heatmap >= thres).nonzero().transpose(0,1);
    vector<torch::Tensor> xyz = tmp_loc.split(1,0);
    torch::Tensor z = heatmap.index(xyz);
    xyz.push_back(z);
//    torch::Tensor ptx = torch::zeros({loc.size(0), loc.size(1)+1});
//    for (int i = 0; i < loc.size(0) ; i++) {
//        ptx[i][0] = loc[i][0];
//        ptx[i][1] = loc[i][1];
//        ptx[i][2] = heatmap[loc[i][0]][loc[i][1]];
//    }
//    cout << ptx;
//    cout << heatmap.index_select(2,loc) << endl;;

    return 0;
}