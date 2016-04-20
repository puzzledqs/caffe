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
void TripletSamplingLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    this->triplet_num_ = this->layer_param_.triplet_sampling_param().triplet_num();
    CHECK_GT(this->triplet_num_, 0);
    LOG(INFO) << "triplet number: " << this->triplet_num_;
}

template <typename Dtype>
void TripletSamplingLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    CHECK_EQ(bottom.size(), 1);
    CHECK_EQ(top->size(), 1);
    CHECK(bottom[0]->num() % 2 == 0);
    int M = bottom[0]->num() / 2;
    int dim = bottom[0]->count() / bottom[0]->num();
    CHECK_LT(this->triplet_num_, M*(M-1)) << " Too many triplets!";
    LOG(INFO) << "pair number: " << M;
    LOG(INFO) << "Dim: " << dim;
    blob1_.Reshape(M, dim, 1, 1);
    blob2_.Reshape(M, dim, 1, 1);
    blob3_.Reshape(M, M, 1, 1);
    (*top)[0]->Reshape(this->triplet_num_, 3, 1, 1);
}

template <typename Dtype>
void TripletSamplingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    CHECK_EQ(bottom.size(), 1);
    int M = bottom[0]->num() / 2;
    int dim = bottom[0]->count() / bottom[0]->num();
    const Dtype* data = bottom[0]->cpu_data();

    /* copy the data from bottom to buffers */
    Dtype* data1 = blob1_.mutable_cpu_data();
    Dtype* data2 = blob2_.mutable_cpu_data();
    for (int i = 0; i < M; i++) {
        //caffe_set(dim, Dtype(i), data1 + blob1_.offset(i));
        //caffe_set(dim, Dtype(i), data2 + blob2_.offset(i));
        caffe_copy(dim, data + bottom[0]->offset(i*2),
                        data1 + blob1_.offset(i));
        caffe_copy(dim, data + bottom[0]->offset(i*2 + 1),
                        data2 + blob2_.offset(i));
    }

    /* Compute the distance matrix.
     *  Assuming the data are L2-normalized by the previous layer, we only
     *  need to consider their inner-product.
     */
    Dtype* data3 = blob3_.mutable_cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, M, dim, (Dtype)1.,
      data1, data2, (Dtype)0., data3);
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         printf("%f ", blob3_.data_at(i, j, 0, 0));
    //     }
    //     printf("\n");
    //     fflush(stdout);
    // }
    // getchar();

    /* Iteratively get hard negative pairs (pairs of different items but has
     *  small distance (large inner-product)
     */
    Dtype *triplet_ind = (*top)[0]->mutable_cpu_data();
    int cnt = 0;
    while (cnt < this->triplet_num_)
    {
        int i = cnt % M;
        Dtype* ptr = data3 + blob3_.offset(i);
        Dtype tmp = *(ptr + i);
        *(ptr + i) = Dtype(-1.f);  // turn off pair of the same item
        Dtype max_v = Dtype(-1.f);
        int max_j = -1;
        for (int j = 0; j < M; j++) {
            if (*(ptr + j) < max_v) {
                max_v = *(ptr + j);
                max_j = j;
            }
        }
        *(ptr + max_j) = Dtype(-1.f); // turn off max_j

        triplet_ind[0] = 2 * i;
        triplet_ind[1] = 2 * i + 1;
        triplet_ind[2] = 2 * max_j + 1;
        triplet_ind += 3;
        cnt++;
    }
}

INSTANTIATE_CLASS(TripletSamplingLayer);

}  // namespace caffe
