#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>

using namespace std;
using namespace cv;

cv::String modelPath = "../models/superpoint_v1_320x240.pt";
cv::String img_path = "../icl_snippet/250.png";

void nms_fast( const vector<at::Tensor> & yx, const at::Tensor & heat_vals, int H, int W, int dist_thresh)
{
    int pad = dist_thresh;
    int nms_cell_size = pad*2+1;

    Mat grid = Mat::zeros(H + 2*pad, W + 2*pad, CV_32S);
    Mat inds = Mat::zeros(H, W, CV_32S);

    auto sorted_rs = heat_vals.sort(0, true); // Tuple ...
    auto sorted_indices = std::get<1>(sorted_rs);
    auto sorted_values = std::get<0>(sorted_rs);
    cout << sorted_indices.sizes() << endl;

    // Check for edge case of 0 or 1 corners.
    int num_indicies = sorted_indices.size(0);
    if(num_indicies == 0){
//        return torch::zeros({3,0}, torch::kInt);
    }
    if(num_indicies == 1){
//        return torch::zeros({3,0}, torch::kInt);
    }

    std::vector<int> vec_x_unordered(yx[1].data<int>(), yx[1].data<int>() + yx[1].numel());
    std::vector<int> vec_y_unordered(yx[0].data<int>(), yx[0].data<int>() + yx[0].numel());
    std::vector<float> vec_value_unodered(heat_vals.data<float>(), heat_vals.data<float>() + heat_vals.numel());

    std::vector<int64> vec_indices(sorted_indices.data<int64>(), sorted_indices.data<int64>() + sorted_indices.numel());
    std::vector<int> vec_x(num_indicies);
    std::vector<int> vec_y(num_indicies);

    for (int i = 0; i < num_indicies; i++){
        vec_y[i] = vec_y_unordered[vec_indices[i]];
        vec_x[i] = vec_x_unordered[vec_indices[i]];
    }

    std::vector<float> vec_value(sorted_values.data<float>(), sorted_values.data<float>() + sorted_values.numel());

//    for (int i = 0; i < num_indicies; i++) {
//        cout << "LOC #" << vec_indices[i] << ": " << vec_value[i] << " [" << vec_y[i] << " - " << vec_x[i] << "] \n";
//    }

//    Mat grid_debug = Mat::zeros(H + 2*pad, W + 2*pad, CV_8UC3);
//   Initialize the grid.
    for (int i = 0; i < num_indicies; i++){
        grid.at<int>(vec_y[i]+pad,vec_x[i]+pad) = 1;
        inds.at<int>(vec_y[i],vec_x[i]) = i;

//        grid_debug.at<Vec3b>(vec_y[i]+pad,vec_x[i]+pad)= Vec3b(0,0,255);
    }

//    cv::imshow("grid_debug", grid_debug);
//    cv::waitKey();


    int count = 0;
//    cout << grid.rows << "-" << grid.cols << endl;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i]+pad, vec_y[i] + pad);
        if(grid.at<int>(pt) == 1){
            cv::Rect roi = cv::Rect(vec_x[i], vec_y[i], nms_cell_size, nms_cell_size);
//            cout << "ROI [" << roi.x << "-" << roi.y << "] \n";
//            cout << "ROI [" << roi.tl() << "] \n";
            grid(roi).setTo(0);
            grid.at<int>(pt) = -1;
            count += 1;
        }
//        cout << "VAL #" << loc << ": " << vec_heat_vals[loc] << " [" << vec_y[loc] << " - " << vec_x[loc] << "] \n";
    }

    cout << count;
}

torch::Tensor imgFile_to_tensor(cv::String file, int img_H, int img_W)
{
    cout << "Img = " << file << endl;
    cv::Mat im = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (im.empty()) return torch::Tensor();
    cv::resize(im, im, Size(img_W, img_H));

    cv::imshow("input",im);
    cv::waitKey(30);

    auto options = torch::TensorOptions().dtype(at::kByte).requires_grad(false);
    torch::Tensor tensor_image = torch::from_blob(im.data, {1, im.rows, im.cols, 1}, options);

    tensor_image = tensor_image.to(at::kFloat)/255.f;
    tensor_image = at::transpose(tensor_image, 1, 2);
    tensor_image = at::transpose(tensor_image, 1, 3);

    return  tensor_image;
}

int main()
{
    int W = 320;
    int H = 240;
    int cell = 8;
    float thres = 0.015f;
    int dist_thresh = 4;

    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelPath);
    assert(module != nullptr);
    std::cout << "cpu ok \n";
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
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
    at::Tensor semi = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();
    at::Tensor coarse_desc = outputs->elements()[1].toTensor().to(at::kCPU).squeeze();

    at::Tensor dense = semi.exp();
    dense = dense / (at::sum(dense,0) + .00001);

    at::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);
    cout << nodust.sizes() << endl;

    int Hc = int(H / cell);
    int Wc = int(W / cell);

    at::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});

    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({Hc*cell, Wc*cell});
    cout << heatmap.sizes() << endl;

    Mat tmp = Mat(H,W, CV_32F, heatmap.data_ptr());

    Mat tmp_f, tmp_g, tmp_c;
    tmp_f = tmp*255;
    tmp_f.convertTo(tmp_g,CV_8UC1);
    applyColorMap(tmp_g, tmp_c, COLORMAP_JET);
    cv::imshow("heatmap", tmp_c); cv::waitKey();

    at::Tensor pts = (heatmap >= thres).nonzero();
//    cout << pts << endl;
    vector<at::Tensor> yx = pts.split(1,1);
    at::Tensor z = heatmap.index(yx).squeeze();
    yx[0] = yx[0].squeeze().to(at::kInt); // Becareful: Convert for std vector casting
    yx[1] = yx[1].squeeze().to(at::kInt);

    double e1 = getTickCount();
    nms_fast(yx,z,H,W, dist_thresh);

    double e2 = getTickCount();
    cout << "\n-NMS-time = " <<(e2-e1)/getTickFrequency();

    cv::destroyAllWindows();
//    nms_fast(xy, z, H, W, dist_thresh);

//    nms_fast(nonZeroCoordinates, heat_val, H, W, dist_thresh);

//    cout << heatmap << endl;
//    cv::Mat tmp = cv::Mat(H,W, CV_32F, heatmap.data_ptr());
//    cv::imshow("test", heatmap_cv); cv::waitKey(); cv::destroyAllWindows();
}