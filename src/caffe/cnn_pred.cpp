#include "caffe/caffe.hpp"
#include <iomanip>
#include <iostream>
#include <gflags/gflags.h>
#include "caffe/cnn_pred.hpp"

using namespace caffe;
using namespace std;

namespace CnnPred {

static shared_ptr<Net<float> > feature_extraction_net_;
static bool isInit_ = false;

void InitCnnModel(string def, string model, string mean, DeviceType dev_type, int dev_id) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 3;
    if (dev_type == GPU) {
        std::cout << "Running on GPU" << std::endl;
        Caffe::SetDevice(dev_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else {
        std::cout << "Running on CPU" << std::endl;
        Caffe::set_mode(Caffe::CPU);
    }
    Caffe::set_phase(Caffe::TEST);

    NetParameter param;
    NetParameterMMLab param_mmlab;
    ReadProtoFromBinaryFileOrDie(def, &param_mmlab);
    Net<float>::TransformNetParameter(&param, param_mmlab);
    //ReadProtoFromTextFileOrDie(def, &param);
    param.mutable_layers(0)->mutable_transform_param()->set_mean_file(mean);
    feature_extraction_net_.reset(new Net<float>(param));
    feature_extraction_net_->CopyTrainedLayersFrom(model);
    std::cout << "Cnn model initilized" << std::endl;
    isInit_ = true;
}

void InitCnnModelTxt(string def, string model, string mean, DeviceType dev_type, int dev_id) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 3;
    if (dev_type == GPU) {
        std::cout << "Running on GPU" << std::endl;
        Caffe::SetDevice(dev_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else {
        std::cout << "Running on CPU" << std::endl;
        Caffe::set_mode(Caffe::CPU);
    }
    Caffe::set_phase(Caffe::TEST);

    NetParameter param;
    ReadProtoFromTextFileOrDie(def, &param);
    param.mutable_layers(0)->mutable_transform_param()->set_mean_file(mean);
    feature_extraction_net_.reset(new Net<float>(param));
    feature_extraction_net_->CopyTrainedLayersFrom(model);
    std::cout << "Cnn model initilized" << std::endl;
    isInit_ = true;
}

void ComputeSingleScore(const char *imagepath, vector<float>* scores) {
    if (!isInit_) {
        LOG(FATAL) << "Cnn is not initilized!";
        return;
    }
    string extract_feature_blob_name("output");
    CHECK(feature_extraction_net_->has_blob(extract_feature_blob_name))
            << "Unknown feature blob name" << extract_feature_blob_name;
    LOG(ERROR) << "Loading image";
    Datum datum;
    vector<Datum> datum_vector;
    ReadImageToDatum(imagepath, 0, 256, 256, &datum);
    datum_vector.push_back(datum);
    boost::static_pointer_cast<MemoryDataLayer<float> >(feature_extraction_net_
            ->layer_by_name("data"))->AddDatumVector(datum_vector);

    LOG(ERROR)<< "Extacting Features";
    vector<Blob<float>*> input_vec; // dummy
    feature_extraction_net_->Forward(input_vec);
    const shared_ptr<Blob<float> > feature_blob = feature_extraction_net_
              ->blob_by_name(extract_feature_blob_name);
    CHECK_EQ(feature_blob->width(), 1) << "Incorrect blob width";
    CHECK_EQ(feature_blob->height(), 1) << "Incorrect blob width";
    int batch_size = feature_blob->num();
    int len = feature_blob->channels();
    scores->clear();
    for (int j = 0; j < len; j++) {
        scores->push_back(feature_blob->data_at(0, j, 0, 0));
    }
}

void ComputeScores(const char *imagelist, vector<vector<float> >* scores) {
    if (!isInit_) {
        LOG(FATAL) << "Cnn is not initilized!";
        return;
    }
    string extract_feature_blob_name("output");
    CHECK(feature_extraction_net_->has_blob(extract_feature_blob_name))
            << "Unknown feature blob name" << extract_feature_blob_name;
    LOG(ERROR) << "Loading images";
    int img_num = boost::static_pointer_cast<ImageDataLayer<float> >(feature_extraction_net_
            ->layer_by_name("data"))->LoadImageList(imagelist);
    LOG(ERROR)<< "Extacting Features";

    Datum datum;
    vector<Blob<float>*> input_vec;

    int start_pos = 0;
    scores->clear();
    while (start_pos < img_num) {
        feature_extraction_net_->Forward(input_vec);
        const shared_ptr<Blob<float> > feature_blob = feature_extraction_net_
              ->blob_by_name(extract_feature_blob_name);
        CHECK_EQ(feature_blob->width(), 1) << "Incorrect blob width";
        CHECK_EQ(feature_blob->height(), 1) << "Incorrect blob width";
        int batch_size = feature_blob->num();
        int len = feature_blob->channels();

        int end_pos = min(start_pos + batch_size, img_num);
        for (int i = 0; i + start_pos < end_pos; i++) {
            vector<float> scr;
            for (int j = 0; j < len; j++)
                scr.push_back(feature_blob->data_at(i, j, 0, 0));
            scores->push_back(scr);
        }
        start_pos = end_pos;
        std::cout << start_pos << " images computed" << std::endl;
    }
    LOG(ERROR)<< "Successfully extracted the features!";
}

void ComputeScores(const vector<cv::Mat> &images, vector<vector<float> >* scores) {
    if (!isInit_) {
        LOG(FATAL) << "Cnn is not initilized!";
        return;
    }
    string extract_feature_blob_name("output");
    CHECK(feature_extraction_net_->has_blob(extract_feature_blob_name))
            << "Unknown feature blob name" << extract_feature_blob_name;

    cout << "Loading images from memory" << endl;
    boost::static_pointer_cast<MemoryDataLayer<float> >(feature_extraction_net_
            ->layer_by_name("data"))->AddCvMat(images);
    LOG(ERROR)<< "Extacting Features";

    vector<Blob<float>*> input_vec;
    scores->clear();
    int num_mini_batches = 1;
    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
        feature_extraction_net_->Forward(input_vec);
        const shared_ptr<Blob<float> > feature_blob = feature_extraction_net_
              ->blob_by_name(extract_feature_blob_name);
        CHECK_EQ(feature_blob->width(), 1) << "Incorrect blob width";
        CHECK_EQ(feature_blob->height(), 1) << "Incorrect blob width";
        int batch_size = feature_blob->num();
        int len = feature_blob->channels();
        for (int i = 0; i < images.size(); i++) {
            vector<float> scr;
            for (int j = 0; j < len; j++) {
                scr.push_back(feature_blob->data_at(i, j, 0, 0));
            }
            scores->push_back(scr);
        }
    }
    LOG(ERROR)<< "Successfully extracted the features!";
}

void ReleaseCnnModel() {
    feature_extraction_net_.reset();
}

}
