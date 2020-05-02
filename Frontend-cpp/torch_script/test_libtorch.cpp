#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include <iostream>

using namespace std;
using namespace cv;


//#define DEBUG

cv::String modelPath = "../models/superpoint_v1_320x240.pt";
cv::String img_path = "../icl_snippet/250.png";

vector<Point> nms_fast( const vector<at::Tensor> & yx, const at::Tensor & heat_vals, int H, int W, int dist_thresh)
{
    int pad = dist_thresh;
    int nms_cell_size = pad*2+1;

    Mat grid = Mat::zeros(H + 2*pad, W + 2*pad, CV_8S);
    Mat inds = Mat::zeros(H, W, CV_32S);

    auto sorted_rs = heat_vals.sort(0, true); // Tuple ...
    auto sorted_indices = std::get<1>(sorted_rs);
    auto sorted_values = std::get<0>(sorted_rs);
    cout << sorted_indices.sizes() << endl;

    // Check for edge case of 0 or 1 corners.
    int num_indicies = sorted_indices.size(0);
    if(num_indicies == 0 || num_indicies == 1){
        vector<Point> dumb(1);
        return dumb;
    }

    vector<int> vec_x_unordered(yx[1].data<int>(), yx[1].data<int>() + yx[1].numel());
    vector<int> vec_y_unordered(yx[0].data<int>(), yx[0].data<int>() + yx[0].numel());
    vector<float> vec_value_unodered(heat_vals.data<float>(), heat_vals.data<float>() + heat_vals.numel());

    vector<int64> vec_indices(sorted_indices.data<int64>(), sorted_indices.data<int64>() + sorted_indices.numel());
    vector<int> vec_x(num_indicies);
    vector<int> vec_y(num_indicies);

    for (int i = 0; i < num_indicies; i++){
        vec_y[i] = vec_y_unordered[vec_indices[i]];
        vec_x[i] = vec_x_unordered[vec_indices[i]];
    }

    vector<float> vec_value(sorted_values.data<float>(), sorted_values.data<float>() + sorted_values.numel());

//    for (int i = 0; i < num_indicies; i++) {
//        cout << "LOC #" << vec_indices[i] << ": " << vec_value[i] << " [" << vec_y[i] << " - " << vec_x[i] << "] \n";
//    }

//   Initialize the grid.
    for (int i = 0; i < num_indicies; i++){
        grid.at<char>(vec_y[i]+pad,vec_x[i]+pad) = 1;
        inds.at<int>(vec_y[i],vec_x[i]) = i;
    }

    int count = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i]+pad, vec_y[i] + pad);
        if(grid.at<char>(pt) == 1){
            cv::Rect roi = cv::Rect(vec_x[i], vec_y[i], nms_cell_size, nms_cell_size);
            grid(roi).setTo(0);
            grid.at<char>(pt) = -1;
            count += 1;
        }
    }

    grid = grid(cv::Rect(pad,pad,W,H));

//    cout << count << " - " << grid.size << endl;
    vector<int> out_x(count), out_y(count), out_indicies(count);
    vector<float> out_val(count);
    vector<Point> rs(count);
    int _store_locate = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i], vec_y[i]);
        if(grid.at<char>(pt) == -1)
        {
            rs[_store_locate] = pt;
//            out_x[_store_locate] = pt.x;
//            out_y[_store_locate] = pt.y;
//            out_val[_store_locate] = vec_value[i];
//            out_indicies[_store_locate] = vec_indices[i];
            _store_locate+=1;

        }
    }

//    for (int j = 0; j < count; j++) {
//        cout << "VAL#" << out_indicies[j] << ": [" << out_x[j] << " - " << out_y[j] << "] = " << out_val[j] << endl;
//    }
    return rs;
}

torch::Tensor imgFile_to_tensor(cv::String file, int img_H, int img_W)
{
    cout << "Img = " << file << endl;
    cv::Mat im = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (im.empty()) return torch::Tensor();
    cv::resize(im, im, Size(img_W, img_H));

#ifdef DEBUG
    cv::imshow("input",im);
    cv::waitKey(30);
#endif

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
    int border_remove = 4;

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
    outputs.release(); // RELEASE to avoid Segmented fault

    at::Tensor dense = semi.exp();
    dense = dense / (at::sum(dense,0) + .00001);

    at::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);
//    cout << nodust.sizes() << endl;

    int Hc = int(H / cell);
    int Wc = int(W / cell);

    at::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});

    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({Hc*cell, Wc*cell});

#ifdef DEBUG
    cout << heatmap.sizes() << endl;
    Mat tmp = Mat(H,W, CV_32F, heatmap.data_ptr());
    Mat tmp_f, tmp_g, tmp_c;
    tmp_f = tmp*255;
    tmp_f.convertTo(tmp_g,CV_8UC1);
    applyColorMap(tmp_g, tmp_c, COLORMAP_JET);
    cv::imshow("heatmap", tmp_c); cv::waitKey();
#endif

    at::Tensor pts = (heatmap >= thres).nonzero();
//    cout << pts << endl;
    vector<at::Tensor> yx = pts.split(1,1);
    at::Tensor z = heatmap.index(yx).squeeze();
    yx[0] = yx[0].squeeze().to(at::kInt); // Becareful: Convert for std vector casting
    yx[1] = yx[1].squeeze().to(at::kInt);

    double e1 = getTickCount();
    vector<Point> pts_nms =  nms_fast(yx,z,H,W, dist_thresh);
    double e2 = getTickCount();
    cout << "FULL-time = " <<(e2-e1)/getTickFrequency() << endl;

    cv::Mat bgr_img = cv::imread(img_path);
    cv::resize(bgr_img, bgr_img, Size(W, H));

    for (int i = 0; i < pts_nms.size() ; i++) {
//        bgr_img.at<Vec3b>(pts_nms[i]) = Vec3b(255,0,0);
        cv::circle(bgr_img, pts_nms[i], 3, Scalar(255,0,0));
    }

    cv::imshow("test", bgr_img); cv::waitKey();
    cv::destroyAllWindows();

//    nms_fast(xy, z, H, W, dist_thresh);
//    nms_fast(nonZeroCoordinates, heat_val, H, W, dist_thresh);
//    cout << heatmap << endl;
//    cv::Mat tmp = cv::Mat(H,W, CV_32F, heatmap.data_ptr());
//    cv::imshow("test", heatmap_cv); cv::waitKey(); cv::destroyAllWindows();
    return 0;
}