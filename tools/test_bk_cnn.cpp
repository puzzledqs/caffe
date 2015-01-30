#include "caffe/bk_cnn.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

using namespace BkCnn;

int main() {
    // string def("overfeat_batch128_ImageDataLayer.mmlab.binproto");
    string def_image("overfeat_batch128_ImageDataLayer.prototxt");
    string def_memory("overfeat_batch128_MemoryDataLayer.prototxt");
    string model("overfeat_iter_5000.caffemodel");
    string mean("imagenet_mean_crop.binaryproto");
    /* imagelist is a text file in which each line contains an image path;
        currently we requrie imagelist contain 128 images, i.e., 128 lines */
    string imagelist("example_data/neg_image_list.txt");
    string imagelist2("example_data/pos_image_list.txt");

    // Initialize cnn model
    BkCnnAPI::InitCnnModel(def_image, model, mean, GPU);

    // Get score for a video (image list)
    std::cout << BkCnnAPI::GetVideoScore(imagelist2.c_str(), NULL) << std::endl;

    // Get scores for each frame (image) in the video (image list)
    float scores[128];
    BkCnnAPI::GetVideoScore(imagelist.c_str(), scores);
    for (int i = 0; i < 10; i++)
        std::cout << "Frame " << i << ": " << scores[i] << std::endl;

    BkCnnAPI::InitCnnModel(def_memory, model, mean, GPU);
    vector<cv::Mat> images;
    std::ifstream infile(imagelist.c_str());
    string filename;
    while (infile >> filename) {
        if (filename.length() > 0) {
            cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
            if (!img.data) {
                std::cout << "could not open/find the image: "
                        << filename << std::endl;
            }
            images.push_back(img);
        }
    }
    infile.close();
    memset(scores, 0, sizeof(scores));
    BkCnnAPI::GetVideoScore(images, scores);
    for (int i = 0; i < 10; i++)
        std::cout << "Frame " << i << ": " << scores[i] << std::endl;
    // free the memory
    BkCnnAPI::ReleaseCnnModel();

    return 0;
}
