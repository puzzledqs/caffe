#include "caffe/bk_cnn.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>

using namespace BkCnn;

int main() {
    // string def("overfeat_batch128_ImageDataLayer.mmlab.binproto");
    string def_image("overfeat_batch128_ImageDataLayer.mmlab.binproto");
    string def_memory("overfeat_batch128_MemoryDataLayer.mmlab.binproto");
    // string model("overfeat_iter_5000.caffemodel");
    string model("overfeat_bk_full_iter_25000.caffemodel");
    string mean("imagenet_mean_crop.binproto");
    /* imagelist is a text file in which each line contains an image path;
        currently we requrie imagelist contain 128 images, i.e., 128 lines */
    string imagelist("example_data/neg_image_list.txt");
    string imagelist2("example_data/pos_image_list.txt");

    // Initialize cnn model
    BkCnnAPI::InitCnnModel(def_image, model, mean, GPU, 0);

    // Get score for a video (image list)
    std::cout << BkCnnAPI::GetVideoScore(imagelist2.c_str(), NULL) << std::endl;

    // Get scores for each frame (image) in the video (image list)
    float scores[128];
    clock_t begin = clock();
    std::cout << BkCnnAPI::GetVideoScore(imagelist.c_str(), scores) << std::endl;
    clock_t end = clock();
    std::cout << end - begin << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << "Frame " << i << ": " << scores[i] << std::endl;

    BkCnnAPI::InitCnnModel(def_memory, model, mean, GPU, 0);
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
    time_t begin1 = time(NULL);
    begin = clock();
    BkCnnAPI::GetVideoScore(images, scores);
    time_t end1 = time(NULL);
    end = clock();
    std::cout << end1 - begin1 << std::endl;
    std::cout << (end - begin)/CLOCKS_PER_SEC << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << "Frame " << i << ": " << scores[i] << std::endl;
    // free the memory
    BkCnnAPI::ReleaseCnnModel();

    return 0;
}
