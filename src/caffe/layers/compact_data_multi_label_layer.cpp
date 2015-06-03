#include <leveldb/db.h>
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

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif


namespace caffe {

template <typename Dtype>
CompactDataMultiLabelLayer<Dtype>::~CompactDataMultiLabelLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void CompactDataMultiLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
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
void CompactDataMultiLabelLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
#ifdef USE_MPI
    LOG(ERROR) << "leveldb with MPI on same node is not yet tested! use at your own risk.";
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
#else
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.data_param().source(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
                           << this->layer_param_.data_param().source() << std::endl
                           << status.ToString();
#endif
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    //LOG(FATAL) << "do not support LMDB at present";
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  string value;
  CvMat mat;
  IplImage *img = NULL;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
      value = this->iter_->value().ToString();
      mat = cvMat(1, 1000 * 1000 * 3, CV_8UC1, const_cast<char *>(value.data()) + sizeof(int));

      //datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
      mat = cvMat(1, 1000 * 1000 * 3, CV_8UC1, (char *)(mdb_value_.mv_data) + sizeof(int));
    //datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  img = cvDecodeImage(&mat, 1);
  // datum size
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

  // load id2label map
  if (this->output_labels_) {
    CHECK(this->layer_param_.data_param().has_label_file()) << "label file is not provided!";
    std::ifstream infile(this->layer_param_.data_param().label_file().c_str());
    int line_cnt, label_len;
    infile >> line_cnt >> label_len;
    CHECK_GT(line_cnt, 0);
    CHECK_GT(label_len, 0);
    string id;
    vector<int> labels(label_len);
    id2label_.clear();
    for (int i = 0; i < line_cnt; i++) {
      infile >> id;
      for (int j = 0; j < label_len; j++)
        infile >> labels[j];
      id2label_[id] = labels;
    }
    infile.close();
    LOG(INFO) << "Loading labels completed";

    CHECK_EQ(this->layer_param_.data_param().label_len_vec_size(),
              top->size() - 1)
            << "label output dimension mismatch";
    int total_label_len = 0;
    label_len_vec_.clear();
    for (int j = 1; j < top->size(); j++) {
      label_len_vec_.push_back(this->layer_param_.data_param().label_len_vec(j-1));
      (*top)[j]->Reshape(this->layer_param_.data_param().batch_size(),
                         this->layer_param_.data_param().label_len_vec(j-1),
                         1,
                         1);
      total_label_len += this->layer_param_.data_param().label_len_vec(j-1);
    }
    CHECK_EQ(total_label_len, label_len) << "total label length mismatch!";
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        label_len, 1, 1);
  }
  this->datum_height_ = crop_size;
  this->datum_width_ = crop_size;
  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;
}

template <typename Dtype>
void CompactDataMultiLabelLayer<Dtype>::Forward_cpu(
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
      std::cout << i << ": ";
      for (int j = 1; j < top->size(); j++) {
        caffe_copy(label_len_vec_[j-1],
                   this->prefetch_label_.cpu_data() + offset,
                   (*top)[j]->mutable_cpu_data() + label_len_vec_[j-1] * i);
        std::cout << (*top)[j]->data_at(i, 0, 0, 0) << " ";
        offset += label_len_vec_[j-1];
      }
      std::cout << std::endl;
    }
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

template <typename Dtype>
void CompactDataMultiLabelLayer<Dtype>::Forward_gpu(
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


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void CompactDataMultiLabelLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  string value;
  CvMat mat;
  IplImage *img = NULL;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

#ifndef USE_MPI
  for (int item_id = 0; item_id < batch_size; ++item_id) {
#else
      for (int item_id = batch_size * Caffe::mpi_self_rank() * (-1); item_id < batch_size * (Caffe::mpi_all_rank() - Caffe::mpi_self_rank()); ++item_id) {
//        For MPI usage, we collectively read batch_size * all_proc samples. Every process will use its
//        own part of samples. This method is more cache and hard disk efficient compared to dataset splitting.
      bool do_read = (item_id>=0) && (item_id<batch_size);
      if(do_read){
#endif
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      value = iter_->value().ToString();
      mat = cvMat(1, 1000 * 1000, CV_8UC1, const_cast<char *>(value.data()) + sizeof(int));

      // datum.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      //LOG(FATAL) << "LMDB is not supported at present";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      mat = cvMat(1, 1000 * 1000 * 3, CV_8UC1, (char *)(mdb_value_.mv_data) + sizeof(int));
      // datum.ParseFromArray(mdb_value_.mv_data,
      //     mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    img = cvDecodeImage(&mat, 1);
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, img, this->mean_, top_data);
    cvReleaseImage(&img);  // release current image
    if (this->output_labels_) {
      // look up id2label to set the labels
      switch(this->layer_param_.data_param().backend()) {
        case DataParameter_DB_LEVELDB:
          top_label[item_id] = *((int *)const_cast<char *>(value.data()));
          break;
        case DataParameter_DB_LMDB: {
          char *tmp = (char *)(mdb_key_.mv_data);
          string id = string(tmp, tmp + mdb_key_.mv_size);
          const vector<int> labels = id2label_[id];
          //std::cout << id << " ";
          int label_len = labels.size();
          CHECK_GT(label_len, 0) << "labels not found for " << id;
          for (int j = 0; j < label_len; j++) {
            top_label[item_id * label_len + j] = (float)labels[j];
            //std::cout << labels[j];
          }
          //std::cout << std::endl;
          break;
        }
        default:
          LOG(FATAL) << "Unkown database backend";
      }
      // LOG(INFO) << "label: " << top_label[item_id];
    }
#ifdef USE_MPI
    }
    else{
//          LOG(INFO)<<" Skipped: "<<item_id<<" "<<(char*)mdb_key_.mv_data;
        }
#endif
    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}


INSTANTIATE_CLASS(CompactDataMultiLabelLayer);

}  // namespace caffe
