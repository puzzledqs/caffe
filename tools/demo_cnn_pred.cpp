#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <utility>
#include <caffe/cnn_pred.hpp>

using namespace std;
using namespace CnnPred;

bool pairComp(const pair<float, int>& p1, const pair<float, int>& p2) {
    return p1.first > p2.first;
}

int main() {
    string model_def = "/home/common/caffe-newversion/examples/imagenet/googlenet_memory_data.prototxt";
    string model = "/home/common/caffe-newversion/models/googlenet/bvlc_googlenet_iter_1180000.caffemodel";
    string mean = "/home/common/caffe-newversion/data/ilsvrc12/imagenet_mean_crop.binaryproto";

    char *testImagePath = "/r15cflmw-home/common/image-cls-demo/book.jpg";
    char *labelFile ="/home/sqiu/dataset/ILSVRC2012/synset_words.txt";

    vector<string> labels;
    string label;
    ifstream infile(labelFile);
    while (!infile.eof()) {
        getline(infile, label);
        labels.push_back(label);
    }
    //infile.close();

    InitCnnModelTxt(model_def, model, mean);

    vector<float> scores;

    ComputeSingleScore(testImagePath, &scores);

    ReleaseCnnModel();
    vector<pair<float, int> > results;
    for (int i = 0; i < scores.size(); i++)
        results.push_back(pair<float, int>(scores[i], i));
    sort(results.begin(), results.end(), pairComp);

    for (int i = 0; i < 5; i++)
        cout << labels[results[i].second] << ": "
                << results[i].first << endl;

}
