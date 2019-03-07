#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

cv::String modelPath = "../models/superpoint_320x320.pt";
cv::String path = "../icl_snippet/*.png";

int main() 
{
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);
  assert(module != nullptr);
  std::cout << "cpu ok \n";
  module->to(at::kCUDA);
  std::cout <<  "gpu ok \n";

  vector<cv::String> fn;
  cv::glob(path,fn,true);
  
  for (size_t k=0; k<fn.size(); ++k)
  {
      cv::Mat im = cv::imread(fn[k]);
      if (im.empty()) continue; //only proceed if sucsessful
      // you probably want to do some preprocessing
      // data.push_back(im);
      cv::imshow("test", im);
      cv::waitKey();
  }
}