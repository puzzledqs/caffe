#include <stdint.h>

#include <string>
#include <vector>

#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CompactTripletCacheDataLayer<Dtype>::~CompactTripletCacheDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  cache_.reset();
}

template <typename Dtype>
void CompactTripletCacheDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
#ifdef USE_MPI
  LOG(ERROR) << "CompactCacheDataLayer doesn't support MPI";
#endif
  this->output_labels_ = false;
  DataLayerSetUp(bottom, top);
  // The subclasses should setup the datum channels, height and width
  CHECK_GT(this->datum_channels_, 0);
  CHECK_GT(this->datum_height_, 0);
  CHECK_GT(this->datum_width_, 0);
  CHECK(this->transform_param_.crop_size() > 0);
  CHECK_GE(this->datum_height_, this->transform_param_.crop_size());
  CHECK_GE(this->datum_width_, this->transform_param_.crop_size());
  int crop_size = this->transform_param_.crop_size();

  CHECK(this->transform_param_.has_mean_file());
  this->data_mean_.Reshape(1, this->datum_channels_, crop_size, crop_size);
  const string& mean_file = this->transform_param_.mean_file();
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  this->data_mean_.FromProto(blob_proto);
  Blob<Dtype> tmp;
  tmp.FromProto(blob_proto);
  const Dtype* src_data = tmp.cpu_data();
  Dtype* dst_data = this->data_mean_.mutable_cpu_data();
  CHECK_EQ(tmp.num(), 1);
  CHECK_EQ(tmp.channels(), this->datum_channels_);
  CHECK_GE(tmp.height(), crop_size);
  CHECK_GE(tmp.width(), crop_size);
  int w_off = (tmp.width() - crop_size) / 2;
  int h_off = (tmp.height() - crop_size) / 2;
  for (int c = 0; c < this->datum_channels_; c++) {
    for (int h = 0; h < crop_size; h++) {
      for (int w = 0; w < crop_size; w++) {
        int src_idx = (c * tmp.height() + h + h_off) * tmp.width() + w + w_off;
        int dst_idx = (c * crop_size + h) * crop_size + w;
        dst_data[dst_idx] = src_data[src_idx];
      }
    }
  }

  this->mean_ = this->data_mean_.cpu_data();
  this->data_transformer_.InitRand();

  this->prefetch_data_.mutable_cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void CompactTripletCacheDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  /* Initialize cache_ */
  LOG(INFO) << "Opening cache " << this->layer_param_.data_param().source();
  cache_.reset(new ImageDataCache());
  bool ret = cache_->LoadCache(this->layer_param_.data_param().source());
  CHECK(ret) << "Failed to load cache "
                << this->layer_param_.data_param().source() << std::endl;

  /* Load triplet list */
  CHECK(this->layer_param_.data_param().has_label_file()) << "label file is not provided!";
  std::ifstream infile(this->layer_param_.data_param().label_file().c_str());
  triplet_vec_.clear();
  string query, similar, dissimilar;
  while (infile >> query >> similar >> dissimilar) {
    vector<string> tmp;
    tmp.push_back(query);
    tmp.push_back(similar);
    tmp.push_back(dissimilar);
    triplet_vec_.push_back(tmp);
  }
  infile.close();
  LOG(INFO) << "Loading triplets completed: " << triplet_vec_.size();

  triplet_cur_ = 0;
  triplet_num_ = triplet_vec_.size();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      triplet_cur_++;
      if (triplet_cur_ == triplet_num_)
        triplet_cur_ = 0;
    }
  }

  // image triplet
  int crop_size = this->layer_param_.transform_param().crop_size();
  this->datum_channels_ = 3;
  CHECK_GT(crop_size, 0) << "crop size must be greater than 0";
  for (int i = 0; i < (*top).size(); i++) {
    (*top)[i]->Reshape(this->layer_param_.data_param().batch_size(),
                       this->datum_channels_ , crop_size, crop_size);
  }
  this->prefetch_data_.Reshape(3 * this->layer_param_.data_param().batch_size(),
        this->datum_channels_ , crop_size, crop_size);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  this->datum_height_ = crop_size;
  this->datum_width_ = crop_size;
  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void CompactTripletCacheDataLayer<Dtype>::InternalThreadEntry() {
  IplImage *img = NULL;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a image blob
    //std::cout << "Triplet #" << item_id << ": ";
    for (int j = 0; j < 3; j++) {
      img = cache_->GetImage(triplet_vec_[triplet_cur_][j]);
      // Apply data transformations (mirror, scale, crop...)
      //std::cout << triplet_vec_[triplet_cur_][j] << ' ';
      this->data_transformer_.Transform(item_id + j * batch_size,
                                         img,
                                         this->mean_,
                                         top_data
                                         );
      cvReleaseImage(&img);  // release current image
    }
    //std::cout << std::endl;
    /* move the image index */
    triplet_cur_++;
    if (triplet_cur_ == triplet_num_)
        triplet_cur_ = 0;
  }
}

template <typename Dtype>
void CompactTripletCacheDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int i = 0; i < (*top).size(); i++) {
    caffe_copy((*top)[i]->count(),
               this->prefetch_data_.cpu_data()
                   + this->prefetch_data_.offset(batch_size*i),
               (*top)[i]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

template <typename Dtype>
void CompactTripletCacheDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  const int batch_size = this->layer_param_.data_param().batch_size();
  for (int i = 0; i < (*top).size(); i++) {
    caffe_copy((*top)[i]->count(),
              this->prefetch_data_.cpu_data()
                   + this->prefetch_data_.offset(batch_size*i),
             (*top)[i]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(CompactTripletCacheDataLayer);

}  // namespace caffe
