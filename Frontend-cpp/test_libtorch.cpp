#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

#include <cudnn.h>

#include <iostream>

using namespace std;
using namespace cv;

cv::String modelPath = "../models/superpoint_320x320.pt";
cv::String path = "../icl_snippet/*.png";

torch::Tensor nms_fast( const torch::Tensor& in_corners, int H, int W, float dist_thresh)
{
    torch::Tensor grid = torch::zeros({H,W}, torch::kInt64);
    torch::Tensor inds = torch::zeros({H,W}, torch::kInt64);

    torch::Tensor rs;
    return rs;
}

int main()
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

    float thres = 0.015f;

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

int main_test_superpoint()
{
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);
    assert(module != nullptr);
    std::cout << "cpu ok \n";
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        device = torch::Device(torch::kCUDA);
        module->to(device);
        std::cout <<  "gpu ok \n";
    }


    vector<cv::String> fn;
    cv::glob(path,fn,true);

    for (size_t k=0; k<fn.size(); ++k)
    {
        cv::Mat im = cv::imread(fn[k]);
        if (im.empty()) continue; //only proceed if sucsessful
        // you probably want to do some preprocessing
        // data.push_back(im);
        resize(im, im, Size(320, 320));
        Mat im_tensor;
        cvtColor(im, im_tensor, cv::COLOR_BGR2GRAY);

        torch::Tensor tensor_image = torch::from_blob(im_tensor.data, {1, im_tensor.rows, im_tensor.cols, 1}, at::kByte);
        tensor_image = tensor_image.to(at::kFloat);
        tensor_image = at::transpose(tensor_image, 1, 2);
        tensor_image = at::transpose(tensor_image, 1, 3);
        cout<<tensor_image.sizes() <<endl;
        torch::Tensor tensor_gpu = tensor_image.cuda();

//        if(tensor_gpu.is_cuda()){cout <<"i see u \n";}

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(tensor_image.cuda());

        auto outputs = module->forward(inputs).toTuple();
        torch::Tensor semi = outputs->elements()[0].toTensor().to(at::kCPU).squeeze(0);
        torch::Tensor coarse_desc = outputs->elements()[1].toTensor().to(at::kCPU).squeeze(0);

        if(semi.is_cuda() && coarse_desc.is_cuda()){
            cout <<"i see u \n";
        } else {
//            torch::save(semi, "semi.pt");
//            torch::save(coarse_desc, "coarse_desc.pt");
        }
        cout << semi.sizes() << "  " << coarse_desc.sizes() << "\n";
//        cout << semi.data();
//        out1.slice()

        cv::imshow("test", im_tensor);
        cv::waitKey();
    }
}