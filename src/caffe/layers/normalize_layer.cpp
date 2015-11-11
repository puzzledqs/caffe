#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define EPSILON 1e-6

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  for (int i = 0; i < num; ++i) {
    Dtype normsqr
            = caffe_cpu_dot(dim, bottom_data + i*dim, bottom_data + i*dim);
    normsqr += EPSILON;
    caffe_cpu_scale<Dtype>(dim, std::pow(normsqr, -0.5),
                    bottom_data + i*dim,
                    top_data + i*dim);
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  for (int i = 0; i < num; ++i) {
    // $\sum{x'_k*d_k}$
    Dtype a = caffe_cpu_dot(dim, top_data + i*dim, top_diff + i*dim);
    // $x'_j*\sum{x'_k*d_k}$
    caffe_cpu_scale(dim, a, top_data + i*dim, bottom_diff + i*dim);
    // $d_j-x'_j*\sum{x'_k*d_k})$
    caffe_sub(dim, top_diff + i*dim, bottom_diff + i*dim, bottom_diff + i*dim);

    /* skip the scaling to avoid diminishing gradients
    // $\sum{x^2_k}$
    a = caffe_cpu_dot(dim, bottom_data+i*d, bottom_data+i*d);
    // $(d_j-x'_j*\sum{x'_k*d_k})) / \sqrt{\sum{x^2_k}}$
    caffe_cpu_scale(dim, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
    */
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);


}  // namespace caffe
