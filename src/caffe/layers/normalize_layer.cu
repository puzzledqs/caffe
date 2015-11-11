#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define EPSILON 1e-6

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype normsqr;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  for (int i = 0; i < num; ++i) {
    caffe_gpu_dot(dim, bottom_data + i*dim, bottom_data + i*dim, &normsqr);
    normsqr += EPSILON;
    caffe_gpu_scale<Dtype>(dim, pow(normsqr, -0.5), bottom_data + i*dim, top_data + i*dim);
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  Dtype a;
  for (int i = 0; i < num; ++i) {
    caffe_gpu_dot(dim, top_data + i*dim, top_diff + i*dim, &a);
    caffe_gpu_scale(dim, a, top_data + i*dim, bottom_diff + i*dim);
    caffe_gpu_sub(dim, top_diff + i*dim,
                    bottom_diff + i*dim,
                    bottom_diff + i*dim);
    /* skip the scaling to avoid diminishing gradients
    caffe_gpu_dot(dim, bottom_data + i*dim, bottom_data + i*dim, &a);
    caffe_gpu_scale(dim, Dtype(pow(a, -0.5)),
                    bottom_diff + i*dim,
                    bottom_diff + i*dim);
    */
  }
}

INSTANTIATE_CLASS(NormalizeLayer);


}  // namespace caffe
