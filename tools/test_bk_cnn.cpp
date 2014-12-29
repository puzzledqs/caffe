#include "caffe/bk_cnn.hpp"
#include <iostream>

using namespace BkCnn;

int main() {
    string def("overfeat_batch128_ImageDataLayer.mmlab.binproto");
    string model("overfeat_iter_5000.caffemodel");
    string mean("imagenet_mean_crop.binaryproto");
    /* imagelist is a text file in which each line contains an image path;
        currently we requrie imagelist contain 128 images, i.e., 128 lines */
    string imagelist("test_image_list_v2.txt");
    string imagelist2("test_image_list_v3.txt");

    // Initialize cnn model
    BkCnnAPI::InitCnnModel(def, model, mean, GPU);

    // Get score for a video (image list)
    std::cout << BkCnnAPI::GetVideoScore(imagelist2.c_str(), NULL) << std::endl;

    // Get scores for each frame (image) in the video (image list)
    float scores[128];
    BkCnnAPI::GetVideoScore(imagelist.c_str(), scores);
    for (int i = 0; i < 10; i++)
        std::cout << "Frame " << i << ": " << scores[i] << std::endl;

    // free the memory
    BkCnnAPI::ReleaseCnnModel();

    return 0;
}
