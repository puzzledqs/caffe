#include "caffe/caffe.hpp"
#include <iomanip>
#include <iostream>
#include <gflags/gflags.h>
#include "caffe/bk_cnn.hpp"

using namespace caffe;
using namespace std;

namespace BkCnn {

// class BkCnn {
//     public:
//         static BkCnn* GetInstance() {
//             if (
//             return instance_;
//         }
//         void Init(string def, string model, string mean, DeviceType dev_type, int dev_id);
//         void GetScores(string imagelist);


// };

static shared_ptr<Net<float> > feature_extraction_net_;
static bool isInit_ = false;

static void Init(string def, string model, string mean, DeviceType dev_type, int dev_id) {
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
    param.mutable_layers(0)->mutable_transform_param()->set_mean_file(mean);
    feature_extraction_net_.reset(new Net<float>(param));
    feature_extraction_net_->CopyTrainedLayersFrom(model);
    std::cout << "Cnn model initilized" << std::endl;
    isInit_ = true;
}

static float GetImageScores(const char *imagelist, float *scores) {
    if (!isInit_) {
        LOG(FATAL) << "BkCnn is not initilized!";
        return -1.f;
    }
    string extract_feature_blob_name("prob");
    CHECK(feature_extraction_net_->has_blob(extract_feature_blob_name))
            << "Unknown feature blob name" << extract_feature_blob_name;
    int num_mini_batches = 1;
    LOG(ERROR) << "Loading images";
    boost::static_pointer_cast<ImageDataLayer<float> >(feature_extraction_net_
            ->layer_by_name("data"))->LoadImageList(imagelist);
    LOG(ERROR)<< "Extacting Features";

    Datum datum;
    vector<Blob<float>*> input_vec;
    float *res = scores;
    if (res == NULL)
        res = new float[128 * num_mini_batches];
    float *p = res;
    float sum = 0.f;

    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
        feature_extraction_net_->Forward(input_vec);
        const shared_ptr<Blob<float> > feature_blob = feature_extraction_net_
              ->blob_by_name(extract_feature_blob_name);
        int batch_size = feature_blob->num();
        for (int i = 0; i < batch_size; i++) {
            *p = 1 - feature_blob->data_at(i, 0, 0, 0);
            sum += *p++;
        }
    }
    if (scores == NULL)
        delete[] res;
    return sum / (128 * num_mini_batches);

    LOG(ERROR)<< "Successfully extracted the features!";
}



void BkCnnAPI::InitCnnModel(string def, string model, string mean,
        DeviceType dev_type, int dev_id) {
    Init(def, model, mean, dev_type, dev_id);
}

float BkCnnAPI::GetVideoScore(const char *imagelist, float *scores) {
    return GetImageScores(imagelist, scores);
}

void BkCnnAPI::ReleaseCnnModel(){
    feature_extraction_net_.reset();
}

}
