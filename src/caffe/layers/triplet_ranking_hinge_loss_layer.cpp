// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

using std::max;

template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    (*top)[1]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* query_data = bottom[0]->cpu_data();
  const Dtype* similar_sample_data = bottom[1]->cpu_data();
  const Dtype* dissimilar_sample_data = bottom[2]->cpu_data();
  Dtype* loss = (*top)[0]->mutable_cpu_data();
  Dtype* acc = (*top)[1]->mutable_cpu_data();

  Dtype* similar_sample_diff = bottom[1]->mutable_cpu_diff();
  Dtype* dissimilar_sample_diff = bottom[2]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  caffe_sub(count, query_data, similar_sample_data,
            similar_sample_diff);
  caffe_sub(count, query_data, dissimilar_sample_data,
            dissimilar_sample_diff);

  loss[0] = Dtype(0);
  acc[0] = Dtype(0);
  Dtype l = Dtype(0);
  float margin
        = this->layer_param_.triplet_ranking_hinge_loss_param().margin();
  Dtype query_similar_distance_norm;
  Dtype query_dissimilar_distance_norm;
  switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) {
      case TripletRankingHingeLossParameter_Norm_L1: {
        for (int i = 0; i < num; ++i) {
          query_similar_distance_norm = caffe_cpu_asum(
              dim, similar_sample_diff + bottom[1]->offset(i));
          query_dissimilar_distance_norm = caffe_cpu_asum(
              dim, dissimilar_sample_diff + bottom[2]->offset(i));
          l = max(Dtype(0), query_similar_distance_norm -
                      query_dissimilar_distance_norm + margin);
          loss[0] += l;
          l = (l > Dtype(0));
          acc[0] += (1 - l);
          caffe_scal(dim, l, similar_sample_diff + bottom[1]->offset(i));
          caffe_scal(dim, l, dissimilar_sample_diff + bottom[2]->offset(i));
        }
        acc[0] /= num;
        break;
      }
      case TripletRankingHingeLossParameter_Norm_L2: {
        for (int i = 0; i < num; ++i) {
          query_similar_distance_norm = caffe_cpu_dot(
              dim, similar_sample_diff + bottom[1]->offset(i),
              similar_sample_diff + bottom[1]->offset(i));
          query_dissimilar_distance_norm = caffe_cpu_dot(
              dim, dissimilar_sample_diff + bottom[2]->offset(i),
              dissimilar_sample_diff + bottom[2]->offset(i));
          l = max(Dtype(0), query_similar_distance_norm -
                      query_dissimilar_distance_norm + margin);
          // std::cout << "Triplet #" << i << ": "
          //           << query_similar_distance_norm << ", "
          //           << query_dissimilar_distance_norm  << ", ";
          // std::cout << (l > Dtype(0)) << std::endl;
          loss[0] += l;
          l = (l > Dtype(0));
          acc[0] += (1 - l);
          caffe_scal(dim, l, similar_sample_diff + bottom[1]->offset(i));
          caffe_scal(dim, l, dissimilar_sample_diff + bottom[2]->offset(i));
        }
        acc[0] /= num;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " <<
            this->layer_param_.triplet_ranking_hinge_loss_param().norm();
      }
  }
}

template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    Dtype* query_sample_diff = (*bottom)[0]->mutable_cpu_diff();
    Dtype* similar_sample_diff = (*bottom)[1]->mutable_cpu_diff();
    Dtype* dissimilar_sample_diff = (*bottom)[2]->mutable_cpu_diff();

    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();

    switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) {
        case TripletRankingHingeLossParameter_Norm_L1: {
          caffe_cpu_sign(count, similar_sample_diff, similar_sample_diff);
          caffe_scal(count, Dtype(-loss_weight / num), similar_sample_diff);
          caffe_cpu_sign(count, dissimilar_sample_diff, dissimilar_sample_diff);
          caffe_scal(count, Dtype(loss_weight / num), similar_sample_diff);
          caffe_sub(count, dissimilar_sample_diff, similar_sample_diff,
                    query_sample_diff);
          break;
        }
        case TripletRankingHingeLossParameter_Norm_L2: {
          caffe_scal(count, Dtype(-2*loss_weight / num), similar_sample_diff);
          caffe_scal(count, Dtype(2*loss_weight / num), dissimilar_sample_diff);
          caffe_cpu_scale(count, Dtype(-1), dissimilar_sample_diff, query_sample_diff);
          caffe_sub(count, query_sample_diff, similar_sample_diff,
                    query_sample_diff);
          break;
        }
        default: {
          LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " <<
              this->layer_param_.triplet_ranking_hinge_loss_param().norm();
        }
    }
  }
}

INSTANTIATE_CLASS(TripletRankingHingeLossLayer);

}  // namespace caffe
