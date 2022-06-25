//
// Created by hoangqc on 18/03/2019.
//

#include "Superpoint.h"
#include "anms.h"

using namespace std;
using namespace cv;

void cvImg_to_tensor(const Mat & img, torch::Tensor & inp);
vector<Point> nms_fast( const vector<at::Tensor> & yx, const at::Tensor & heat_vals, int H, int W, int dist_thresh, int _border_remove);
void non_maximum_suppression(const vector<at::Tensor> & yx, const at::Tensor & heat_vals, std::vector<cv::KeyPoint> & kps, int H, int W, int dist_thresh, int _border);
cv::Mat tensor2d_to_mat(at::Tensor & tensor);

Superpoint::Superpoint()
{

}

Superpoint::~Superpoint()
{
}

void Superpoint::init(const cv::String & model_path, bool debug, bool use_cuda)
{
    m_debug = debug;
    m_use_cuda = use_cuda;
    torch::requires_grad(false);

    if (torch::cuda::is_available() && m_use_cuda) {
        m_device_type = torch::kCUDA;
        torch::Device device(torch::kCUDA);
        module = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(model_path, torch::Device(m_device_type)));
        assert(module != nullptr);
        cout <<  "gpu ok \n";
    } else{
        torch::Device device(torch::kCPU);
        module = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(model_path, torch::Device(m_device_type)));
        assert(module != nullptr);
        cout << "cpu ok \n";
    }
}

void Superpoint::clear()
{
    // Do something
    return;
}

void Superpoint::compute_NN(cv::Mat &bgr_img) {
    cv::Mat im_gray, im;
    cv::cvtColor(bgr_img, im_gray, cv::COLOR_BGR2GRAY);
//    cv::resize(im_gray, im, cv::Size(W, H), cv::INTER_LINEAR);
    torch::Tensor inp;
    cvImg_to_tensor(im_gray, inp);

    if(m_use_cuda) inputs.emplace_back(inp.cuda());
    else inputs.emplace_back(inp);

    outputs = module->forward(inputs).toTuple();
    if (m_use_cuda){
        semi = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();
        coarse_desc = outputs->elements()[1].toTensor().to(at::kCPU);
    } else{
        semi = outputs->elements()[0].toTensor().squeeze();
        coarse_desc = outputs->elements()[1].toTensor().squeeze();
    }

    inputs.clear();
//    outputs->elements().empty();outputs->elements().clear();
    outputs.release(); // RELEASE to avoid Segmented fault
}

void Superpoint::getKeyPoints(std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    at::Tensor dense = semi.exp();
    dense = dense / (at::sum(dense,0) + .00001);

    at::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);

    int Hc = int(H / cell);
    int Wc = int(W / cell);

    at::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});
    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({Hc*cell, Wc*cell}); //HxW

    at::Tensor pts = (heatmap >= thres).nonzero();
    vector<at::Tensor> yx = pts.split(1,1);
    pts.transpose(0,1);
    at::Tensor z = heatmap.index({yx[0],yx[1]}).squeeze();

    yx[0] = yx[0].squeeze().to(at::kInt);
    yx[1] = yx[1].squeeze().to(at::kInt);
    
    non_maximum_suppression(yx,z,kps, H, W, dist_thresh, border_remove);

    const long num_pts = kps.size();
    const long D = coarse_desc.size(1);

    vector<float> sample_pts(kps.size()*2);
    float W_2 = float(W)/2.f;
    float H_2 = float(H)/2.f;
    for (int i = 0; i < kps.size() ; i++) {
        sample_pts[i*2] = float(kps[i].pt.x)/W_2 - 1.f;
        sample_pts[i*2+1] = float(kps[i].pt.y)/H_2 - 1.f;
        //Update real scale
//        kps[i].pt = kps[i].pt*scale;
    }

    at::Tensor sample_tensor = torch::from_blob(sample_pts.data(), {1, 1, num_pts, 2}, torch::TensorOptions().dtype(at::kFloat));
    int64_t interpolation_mode = 0; // Bilinear
    int64_t padding_mode = 0; // zeros
    at::Tensor tmp_desc = at::grid_sampler(coarse_desc, sample_tensor, interpolation_mode, padding_mode, false);
    tmp_desc = tmp_desc.reshape({D, -1});

    at::Tensor desc_norm = torch::norm(tmp_desc, 2, 0).unsqueeze(0);// Frobenius norm on channel 0
    tmp_desc = torch::div(tmp_desc, desc_norm);
    desc = tensor2d_to_mat(tmp_desc).t();
}

void Superpoint::run(cv::Mat & bgr_img)
{
    double e1 = getTickCount();

    cv::Mat im_gray, im;
    cv::cvtColor(bgr_img, im_gray, cv::COLOR_BGR2GRAY);
    cv::resize(im_gray, im, cv::Size(W, H), cv::INTER_AREA);

    torch::Tensor inp;
    cvImg_to_tensor(im, inp);

    if(m_use_cuda){
        inputs.emplace_back(inp.cuda());
    } else {
        inputs.emplace_back(inp);
    }

    outputs = module->forward(inputs).toTuple();

    if (m_use_cuda){
        semi = outputs->elements()[0].toTensor().to(at::kCPU).squeeze();
        coarse_desc = outputs->elements()[1].toTensor().to(at::kCPU);
//        cout << coarse_desc.sizes();
    } else{
        semi = outputs->elements()[0].toTensor().squeeze();
        coarse_desc = outputs->elements()[1].toTensor().squeeze();
    }

    inputs.clear();

//    outputs->elements().empty(); outputs->elements().clear();
    outputs.release(); // RELEASE to avoid Segmented fault

    at::Tensor dense = semi.exp();
    dense = dense / (at::sum(dense,0) + .00001);

    at::Tensor nodust = dense.slice(0,0,dense.size(0)-1);
    nodust = nodust.transpose(0,2).transpose(0,1);

    int Hc = int(H / cell);
    int Wc = int(W / cell);

    at::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});

    heatmap = heatmap.transpose(1,2);
    heatmap = heatmap.reshape({Hc*cell, Wc*cell}); //HxW
//    cout << "Heatmap:" << heatmap.sizes() << endl;

    at::Tensor pts = (heatmap >= thres).nonzero();
//    cout << pts.sizes();
    vector<at::Tensor> yx = pts.split(1,1);
//    cout << yx[0] << endl; //    cout << yx[1] << endl;
    pts.transpose(0,1);
    at::Tensor z = heatmap.index({yx[0],yx[1]}).squeeze();

    yx[0] = yx[0].squeeze().to(at::kInt); // Becareful: Convert for std vector casting
    yx[1] = yx[1].squeeze().to(at::kInt);

    std::vector<cv::Point> m_pts_nms =  nms_fast(yx,z,H,W, dist_thresh, border_remove);

    const long num_pts = m_pts_nms.size();
    const long D = coarse_desc.size(1);

    vector<float> sample_pts(m_pts_nms.size()*2);
    float W_2 = float(W)/2.f;
    float H_2 = float(H)/2.f;
    for (int i = 0; i < m_pts_nms.size() ; i++)
    {
        sample_pts[i*2] = float(m_pts_nms[i].x)/W_2 - 1.f;
        sample_pts[i*2+1] = float(m_pts_nms[i].y)/H_2 - 1.f;
    }
    at::Tensor sample_tensor = torch::from_blob(sample_pts.data(), {1, 1, num_pts, 2}, torch::TensorOptions().dtype(at::kFloat));
    int64_t interpolation_mode = 0; // Bilinear
    int64_t padding_mode = 0; // zeros
    m_desc = at::grid_sampler(coarse_desc, sample_tensor, interpolation_mode, padding_mode, false);
    m_desc = m_desc.reshape({D, -1});

    at::Tensor desc_norm = torch::norm(m_desc, 2, 0).unsqueeze(0);// Frobenius norm on channel 0
//    cout << desc_norm.sizes() << endl; //    m_desc /= desc_norm;
    m_desc = torch::div(m_desc, desc_norm);

    double e2 = getTickCount();
//    cout << "Full-time = " <<(e2-e1)/getTickFrequency() << endl;
//    cout << m_pts_nms.size() << endl;

    std::vector<float> v_tmp(m_desc.data_ptr<float>(), m_desc.data_ptr<float>() + m_desc.numel());
    cv::Mat temp(m_desc.size(0), m_desc.size(1),CV_32FC1, v_tmp.data());
    cv::Mat desc = temp.t();

    if(m_debug)
    {
        for (int i = 0; i < m_pts_nms.size() ; i++) {
            cv::circle(bgr_img, m_pts_nms[i]*2, 2, Scalar(255,0,0),2, FILLED);
//            bgr_img.at<cv::Vec3b>(m_pts_nms[i]) = cv::Vec3b(255,0,0);
        }
        cv::imshow("test", bgr_img); //cv::waitKey(1);
//        cv::imshow("desc", desc);
    }


}

void cvImg_to_tensor(const Mat & img, torch::Tensor & inp)
{
    auto options = torch::TensorOptions().dtype(at::kByte).requires_grad(false);
    inp = torch::from_blob(img.data, {1, img.rows, img.cols, 1}, options);
    inp = inp.to(at::kFloat)/255.f;
    inp = at::transpose(inp, 1, 2);
    inp = at::transpose(inp, 1, 3);
}

void non_maximum_suppression(const vector<at::Tensor> & yx, const at::Tensor & heat_vals, std::vector<cv::KeyPoint> & kps, int H, int W, int dist_thresh, int _border)
{
    int pad = dist_thresh;
    int nms_cell_size = pad*2+1;
    Mat grid = Mat::zeros(H + 2*pad, W + 2*pad, CV_8S);
    Mat inds = Mat::zeros(H, W, CV_32S);

    auto sorted_rs = heat_vals.sort(0, true); // Tuple ...
    auto sorted_indices = std::get<1>(sorted_rs);
    auto sorted_values = std::get<0>(sorted_rs);
//    cout << sorted_indices.sizes() << endl;

    // Check for edge case of 0 or 1 corners.
    int num_indicies = sorted_indices.size(0);
    if(num_indicies == 0 || num_indicies == 1){
        cout<< "No Feature Detected!";
        return;
    }
    vector<int> vec_x_unordered(yx[1].data_ptr<int>(), yx[1].data_ptr<int>() + yx[1].numel());
    vector<int> vec_y_unordered(yx[0].data_ptr<int>(), yx[0].data_ptr<int>() + yx[0].numel());
    vector<float> vec_value_unodered(heat_vals.data_ptr<float>(), heat_vals.data_ptr<float>() + heat_vals.numel());
    vector<int64> vec_indices(sorted_indices.data_ptr<int64>(), sorted_indices.data_ptr<int64>() + sorted_indices.numel());
    vector<int> vec_x(num_indicies);
    vector<int> vec_y(num_indicies);

    for (int i = 0; i < num_indicies; i++){
        vec_y[i] = vec_y_unordered[vec_indices[i]];
        vec_x[i] = vec_x_unordered[vec_indices[i]];
    }
    vector<float> vec_value(sorted_values.data_ptr<float>(), sorted_values.data_ptr<float>() + sorted_values.numel());

//   Initialize the grid.
    for (int i = 0; i < num_indicies; i++){
        grid.at<char>(vec_y[i]+pad,vec_x[i]+pad) = char(127);
        inds.at<int>(vec_y[i],vec_x[i]) = i;
    }

    int count = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i]+pad, vec_y[i] + pad);
        if(grid.at<char>(pt) == char(127)){
            cv::Rect roi = cv::Rect(vec_x[i], vec_y[i], nms_cell_size, nms_cell_size);
            grid(roi).setTo(0);
            grid.at<char>(pt) = -127;
            count += 1;
        }
    }

    // Store results + remove border points
    grid = grid(cv::Rect(pad , pad , W , H ));
    cv::Rect inRect = cv::Rect(_border, _border, W - 2*_border, H - 2*_border);

    kps.resize(count);
    int _store_locate = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point2f pt = Point2f(vec_x[i], vec_y[i]);
        if(inRect.contains(pt) && grid.at<char>(pt) == char(-127))
        {
            kps[_store_locate].pt = pt;
            kps[_store_locate].response = vec_value[i];
            _store_locate+=1;
        }
    }
    kps.resize(_store_locate);
}

cv::Mat tensor2d_to_mat(at::Tensor & tensor)
{
    std::vector<float> v_tmp(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return cv::Mat(tensor.size(0), tensor.size(1),CV_32FC1, v_tmp.data());
}

vector<Point> nms_fast( const vector<at::Tensor> & yx, const at::Tensor & heat_vals, int H, int W, int dist_thresh, int _border_remove)
{
    int pad = dist_thresh;
    int nms_cell_size = pad*2+1;

    Mat grid = Mat::zeros(H + 2*pad, W + 2*pad, CV_8S);
    Mat inds = Mat::zeros(H, W, CV_32S);

    auto sorted_rs = heat_vals.sort(0, true); // Tuple ...
    auto sorted_indices = std::get<1>(sorted_rs);
    auto sorted_values = std::get<0>(sorted_rs);
//    cout << sorted_indices.sizes() << endl;

    // Check for edge case of 0 or 1 corners.
    int num_indicies = sorted_indices.size(0);
    if(num_indicies == 0 || num_indicies == 1){
        vector<Point> dumb(1);
        return dumb;
    }

    vector<int> vec_x_unordered(yx[1].data_ptr<int>(), yx[1].data_ptr<int>() + yx[1].numel());
    vector<int> vec_y_unordered(yx[0].data_ptr<int>(), yx[0].data_ptr<int>() + yx[0].numel());
    vector<float> vec_value_unodered(heat_vals.data_ptr<float>(), heat_vals.data_ptr<float>() + heat_vals.numel());

    vector<int64> vec_indices(sorted_indices.data_ptr<int64>(), sorted_indices.data_ptr<int64>() + sorted_indices.numel());
    vector<int> vec_x(num_indicies);
    vector<int> vec_y(num_indicies);

    for (int i = 0; i < num_indicies; i++){
        vec_y[i] = vec_y_unordered[vec_indices[i]];
        vec_x[i] = vec_x_unordered[vec_indices[i]];
    }

    vector<float> vec_value(sorted_values.data_ptr<float>(), sorted_values.data_ptr<float>() + sorted_values.numel());

//    for (int i = 0; i < num_indicies; i++) {
//        cout << "LOC #" << vec_indices[i] << ": " << vec_value[i] << " [" << vec_y[i] << " - " << vec_x[i] << "] \n";
//    }

//   Initialize the grid.

    for (int i = 0; i < num_indicies; i++){
        grid.at<char>(vec_y[i]+pad,vec_x[i]+pad) = char(127);
        inds.at<int>(vec_y[i],vec_x[i]) = i;
    }

    int count = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i]+pad, vec_y[i] + pad);
        if(grid.at<char>(pt) == char(127)){
            cv::Rect roi = cv::Rect(vec_x[i], vec_y[i], nms_cell_size, nms_cell_size);
            grid(roi).setTo(0);
            grid.at<char>(pt) = -127;
            count += 1;
        }
    }

    // Store results + remove border points
    grid = grid(cv::Rect(pad , pad , W , H ));
    cv::Rect inRect = cv::Rect(_border_remove, _border_remove, W - 2*_border_remove, H - 2*_border_remove);

//    cout << count << " - " << grid.size << endl;
    vector<int> out_x(count), out_y(count), out_indicies(count);
    vector<float> out_val(count);
    vector<Point> rs(count);
    int _store_locate = 0;
    for (int i = 0; i < num_indicies; i++) {
        Point pt = Point(vec_x[i], vec_y[i]);
        if(inRect.contains(pt) && grid.at<char>(pt) == char(-127))
        {
            rs[_store_locate] = pt;
//            out_val[_store_locate] = vec_value[i];
//            out_indicies[_store_locate] = vec_indices[i];
            _store_locate+=1;
        }
    }

//    cout << "Filtered : " << _store_locate << endl;
    rs.resize(_store_locate);


    return rs;
}
