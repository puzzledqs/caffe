#ifndef BK_CNN_
#define BK_CNN_
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
using namespace std;


namespace BkCnn {
    enum DeviceType {CPU, GPU};

    class BkCnnAPI {
     public:
        /// \brief Initialize the cnn model
        ///
        /// must be called before GetVideoScore
        /// \param def definition file, can be found under libbkcnn/data
        /// \param model model parameters, can be found under libbkcnn/data
        /// \param mean image mean, can be found under libbkcnn/data
        /// \param dev_type indicates the device this lib is to run on
        /// \param dev_id indicates the GPU id to run on (effective only when dev_type is set to GPU)
        static void InitCnnModel(string def, string model, string mean,
                        DeviceType dev_type = GPU, int dev_id = 0);

        /// \brief Get the bk score (between 0 and 1) of a given video
        ///
        /// now a video is represented by 128 frames, this number is fixed
        /// \param imagelist the text file containing the paths to the image frames extracted from a video. The file is expected to have 128 lines, each being a path to an image frame.
        /// \param scores a pointer to hold the score of each image frame, the memory should be allocated BEFORE passed into this function.
        /// \return output the bk score, which is the average of scores[]
        static float GetVideoScore(const char *imagelist, float *scores);

        /// \brief Get the bk score (between 0 and 1) of a given video
        ///
        /// now a video is represented by image frames: vector<cv::Mat>
        /// \param images  the image frames extracted from a video. The size of images (images.size()) should be <= 128
        /// \param scores a pointer to hold the score of each image frame, the memory should be allocated BEFORE passed into this function.
        /// \return output the bk score, which is the average of scores[]
        static float GetVideoScore(const vector<cv::Mat> &images, float *scores);

        /// \brief Release the memory occupied by the cnn model
        ///
        /// must called before the program exits
        static void ReleaseCnnModel();
    };
}
#endif
