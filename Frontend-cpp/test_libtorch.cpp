#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cudnn.h>

#include <iostream>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace cv;

cv::String modelPath = "../models/superpoint_320x320.pt";
cv::String path = "../icl_snippet/*.png";

//struct Net : torch::nn::Module {
//    Net(int64_t N, int64_t M) {
//        W = register_parameter("W", torch::randn({N, M}));
//        b = register_parameter("b", torch::randn(M));
//    }
//    torch::Tensor forward(torch::Tensor input) {
//        return torch::addmm(b, input, W);
//    }
//    torch::Tensor W, b;
//};

//void convert_to_tensor()
//{
//
//}

int main() 
{
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);
    assert(module != nullptr);
    std::cout << "cpu ok \n";
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
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
        } else{
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