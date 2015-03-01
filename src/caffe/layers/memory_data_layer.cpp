#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.memory_data_param().channels();
  this->datum_height_ = this->layer_param_.memory_data_param().height();
  this->datum_width_ = this->layer_param_.memory_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;
  CHECK_GT(batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  int crop_size = this->layer_param_.transform_param().crop_size();
  CHECK_GT(crop_size, 0) << "crop size must be greater than 0";
  (*top)[0]->Reshape(batch_size_, this->datum_channels_, crop_size,
                     crop_size);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);
  added_data_.Reshape(batch_size_, this->datum_channels_, crop_size,
                      crop_size);
  added_label_.Reshape(batch_size_, 1, 1, 1);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add Datum when earlier ones haven't been consumed"
      << " by the upper layers";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_LE(num, batch_size_) <<
      "The number of added datum must be no greater than the batch size";

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int batch_item_id = 0; batch_item_id < num; ++batch_item_id) {
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(
        batch_item_id, datum_vector[batch_item_id], this->mean_, top_data);
    top_label[batch_item_id] = datum_vector[batch_item_id].label();
  }
  // num_images == batch_size_
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddCvMat(const vector<cv::Mat>& images) {
  CHECK(!has_new_data_) <<
    "Can't add Datum when earlier ones havn't been consumed"
    << " by the upper layers";
  size_t num = images.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_LE(num, batch_size_) <<
    "The number of added datum must be no greater than the batch size";

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int i = 0; i < images.size(); i++) {
    cv::Mat cv_img;
    const cv::Mat& cv_img_origin = images[i];
    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not load data";
        return;
    }
    if (cv_img_origin.rows == this->datum_height_ && cv_img_origin.cols == this->datum_width_)
      this->data_transformer_.Transform(i, cv_img_origin, this->mean_, top_data);
    else {
      cv::resize(cv_img_origin, cv_img, cv::Size(this->datum_height_, this->datum_width_));
      this->data_transformer_.Transform(i, cv_img, this->mean_, top_data);
    }
    top_label[i] = 0;
  }
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  (*top)[0]->set_cpu_data(data_ + pos_ * this->datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
