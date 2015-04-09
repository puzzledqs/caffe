#ifndef CNN_PRED_
#define CNN_PRED_
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
using namespace std;

namespace CnnPred {
    enum DeviceType {CPU, GPU};
    /// \brief Initialize the cnn model
    ///
    /// must be called before GetVideoScore
    /// \param def definition file, can be found under libbkcnn/data
    /// \param model model parameters, can be found under libbkcnn/data
    /// \param mean image mean, can be found under libbkcnn/data
    /// \param dev_type indicates the device this lib is to run on
    /// \param dev_id indicates the GPU id to run on (effective only when dev_type is set to GPU)
    void InitCnnModel(string def, string model, string mean,
                    DeviceType dev_type = CPU, int dev_id = 0);

    void InitCnnModelTxt(string def, string model, string mean,
                    DeviceType dev_type = CPU, int dev_id = 0);

    // compute single image
    void ComputeSingleScore(const char *imagepath, vector<float>* scores);

    // compute arbitrary number of images
    void ComputeScores(const char *imagelist, vector<vector<float> >* scores);

    // compu\te at most one mini_batch, i.e. images.size() <= batch_size
    void ComputeScores(const vector<cv::Mat> &images, vector<vector<float> >* scores);

    /// \brief Release the memory occupied by the cnn model
    ///
    /// must called before the program exits
    void ReleaseCnnModel();
}
#endif
