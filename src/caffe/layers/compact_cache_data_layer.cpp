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
CompactCacheDataLayer<Dtype>::~CompactCacheDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  cache_.reset();
}

template <typename Dtype>
void CompactCacheDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
#ifdef USE_MPI
  LOG(ERROR) << "CompactCacheDataLayer doesn't support MPI";
#endif
  if (top->size() == 1) {
    this->output_labels_ = false;
  } else {
    this->output_labels_ = true;
  }
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
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void CompactCacheDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  /* Initialize cache_ */
  LOG(INFO) << "Opening cache " << this->layer_param_.data_param().source();
  cache_.reset(new ImageDataCache());
  bool ret = cache_->LoadCache(this->layer_param_.data_param().source(), 0, 3);
  CHECK(ret) << "Failed to load cache "
                << this->layer_param_.data_param().source() << std::endl;

  /* Load label */
  if (this->output_labels_) {
    CHECK(this->layer_param_.data_param().has_label_file()) << "label file is not provided!";
    std::ifstream infile(this->layer_param_.data_param().label_file().c_str());
    int line_cnt, label_len;
    infile >> line_cnt >> label_len;
    CHECK_GT(line_cnt, 0);
    CHECK_GT(label_len, 0);
    string id;
    vector<int> labels(label_len);
    image_ids_.clear();
    labels_vec_.clear();
    for (int i = 0; i < line_cnt; i++) {
      infile >> id;
      image_ids_.push_back(id);

      for (int j = 0; j < label_len; j++)
        infile >> labels[j];
      labels_vec_.push_back(labels);
    }
    infile.close();
    LOG(INFO) << "Loading image_ids & labels completed: " << image_ids_.size();

    CHECK_EQ(this->layer_param_.data_param().label_len_vec_size(),
              top->size() - 1)
            << "label output dimension mismatch";
    int specified_label_len = 0;
    label_len_vec_.clear();
    for (int j = 1; j < top->size(); j++) {
      label_len_vec_.push_back(this->layer_param_.data_param().label_len_vec(j-1));
      (*top)[j]->Reshape(this->layer_param_.data_param().batch_size(),
                         this->layer_param_.data_param().label_len_vec(j-1),
                         1,
                         1);
      specified_label_len += this->layer_param_.data_param().label_len_vec(j-1);
    }
    CHECK_EQ(specified_label_len, label_len) << "output label length mismatch!";
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        specified_label_len, 1, 1);
  } /* end load label */

  image_cur_ = 0;
  image_num_ = image_ids_.size();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      image_cur_++;
      if (image_cur_ == image_num_)
        image_cur_ = 0;
    }
  }
  // Read a data point, and use it to initialize the top blob.
  IplImage *img = cache_->GetImage(image_ids_[image_cur_]);
  this->datum_channels_ = img->nChannels;
  cvReleaseImage(&img);

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  CHECK_GT(crop_size, 0) << "crop size must be greater than 0";
  (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                     this->datum_channels_ , crop_size, crop_size);
  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
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
void CompactCacheDataLayer<Dtype>::InternalThreadEntry() {
  IplImage *img = NULL;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a image blob
    img = cache_->GetImage(image_ids_[image_cur_]);
    // Apply data transformations (mirror, scale, crop...)
    //std::cout << image_ids_[image_cur_] << std::endl;
    this->data_transformer_.Transform(item_id, img, this->mean_, top_data);
    cvReleaseImage(&img);  // release current image

    if (this->output_labels_) {
      vector<int> labels = labels_vec_[image_cur_];
      int label_len = labels.size();
      for (int j = 0; j < label_len; j++) {
        top_label[item_id * label_len + j] = (float)labels[j];
        //std::cout << labels[j] << " ";
      }
      //std::cout << std::endl;
    }
    /* move the image index */
    image_cur_++;
    if (image_cur_ == image_num_)
        image_cur_ = 0;
  }
}

template <typename Dtype>
void CompactCacheDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  int offset = 0;
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (this->output_labels_) {
    for (int i = 0; i < batch_size; i++) {
      //std::cout << i << ": ";
      for (int j = 1; j < top->size(); j++) {
        caffe_copy(label_len_vec_[j-1],
                   this->prefetch_label_.cpu_data() + offset,
                   (*top)[j]->mutable_cpu_data() + label_len_vec_[j-1] * i);
        //std::cout << (*top)[j]->data_at(i, 0, 0, 0) << " ";
        offset += label_len_vec_[j-1];
      }
      //std::cout << std::endl;
    }
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

template <typename Dtype>
void CompactCacheDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_gpu_data());
  int offset = 0;
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (this->output_labels_) {
    for (int i = 0; i < batch_size; i++) {
      //std::cout << i << ": ";
      for (int j = 1; j < top->size(); j++) {
        caffe_copy(label_len_vec_[j-1],
                   this->prefetch_label_.cpu_data() + offset,
                   (*top)[j]->mutable_gpu_data() + label_len_vec_[j-1] * i);
        //std::cout << (*top)[j]->data_at(i, 0, 0, 0) << " ";
        offset += label_len_vec_[j-1];
      }
      //std::cout << std::endl;
    }
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(CompactCacheDataLayer);

}  // namespace caffe
