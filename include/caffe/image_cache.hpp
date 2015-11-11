#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>


using namespace std;

namespace caffe {

class BaseDataCache {
 public:
    BaseDataCache();
    virtual ~BaseDataCache() {Release();}
    bool Init(long cache_size);
    void Release();

    bool AppendFile(string filename);
    const char * GetData(int i, long &size) const;
    bool AppendData(void *new_data, long size);

    int GetN() const {return offset_vec_.size() - 1;}
    long GetCacheSize() const {return offset_vec_[GetN()];}
    long GetCacheCapacity() const {return cache_capacity_;}

    void Print();
    void WriteToFile(string filename);

 private:
    char *data_;
    vector<long> offset_vec_;
    long cache_capacity_;
    bool is_init;
};

class ImageDataCache :
    public BaseDataCache {
 public:
    ImageDataCache(): BaseDataCache() {}
    virtual ~ImageDataCache() {}
    void Release();

    void BuildCacheFromImageList(string image_root, string image_list_file, string cache_dir);
    bool LoadCache(string cache_dir, int start = 0, int end = 0);
    bool AppendImageList(string imagelist_file);
    void WriteImageList(string imagelist_file);

    bool AppendImageData(string image_path);
    IplImage* GetImage(string image_name) {return GetImage(GetImageIndex(image_name));}
    void ShowImage(string image_name) {return ShowImage(GetImageIndex(image_name));}
 protected:
    IplImage* GetImage(int i) const;
    void ShowImage(int i) const;
    int GetImageIndex(string image_name)  {return image2id_[image_name];}

 private:
    vector<string> image_list_;
    map<string, int> image2id_;
    string cache_dir_;
    string image_root_;
};

}
