#include "caffe/image_cache.hpp"
#include <boost/filesystem.hpp>
#include <fstream>

#define MAX_FILENAME_LENGTH 256

namespace caffe {

BaseDataCache::BaseDataCache() {
    data_ = NULL;
    cache_capacity_ = 0;
    offset_vec_.push_back(0);
    is_init = false;
}

bool BaseDataCache::Init(long cache_size) {
    if (is_init) {
        cout << "BaseDataCache already initialized!" << endl;
        return false;
    }
    cache_capacity_ = cache_size;
    data_ = new char[cache_capacity_];
    if (data_ != NULL)
        is_init = true;
    return is_init;
}

void BaseDataCache::Release(){
    if (data_ != NULL)
        delete[] data_;
    data_ = NULL;
    cache_capacity_ = 0L;
    offset_vec_.clear();
    offset_vec_.push_back(0);
    is_init = false;
}

bool BaseDataCache::AppendFile(string data_file) {
    FILE *fp = fopen(data_file.c_str(), "rb");
    if (!fp) {
        cout << data_file << " failed" << endl;
        return false;
    }
    int N;
    fread(&N, sizeof(int), 1, fp);
    long *tmp_vec = new long[N + 1];
    fread(tmp_vec, sizeof(long), N + 1, fp);
    long data_size = tmp_vec[N];
    long st = GetCacheSize();
    if (st + data_size > cache_capacity_) {
        cout << data_file << ": cache overflow!" << endl;
        return false;
    }
    fread(data_ + st, sizeof(char), data_size, fp);
    for (int i = 1; i <= N; i++)
        offset_vec_.push_back(st + tmp_vec[i]);
    return true;
}


bool BaseDataCache::AppendData(void *new_data, long size) {
    if (!is_init) {
        cout << "BaseDataCache not initialized!" << endl;
        return false;
    }
    long st = GetCacheSize();
    if (st + size > cache_capacity_) {
        cout << "cache overflow!" << endl;
        return false;
    }
    memcpy((void *)(data_ + st), new_data, size);
    offset_vec_.push_back(st + size);
    return true;
}

const char* BaseDataCache::GetData(int i, long &size) const {
    if (!is_init) {
        cout << "BaseDataCache not initialized!" << endl;
        return NULL;
    }
    if (i < 0 || i >= GetN()) {
        cout << "Invalid subscript: " << i << endl;
        return NULL;
    }
    long st = offset_vec_[i];
    long ed = offset_vec_[i + 1];
    size = ed - st;
    return data_ + st;
}

void BaseDataCache::WriteToFile(string filename) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        cout << filename << " failed" << endl;
        return;
    }
    cout << "writing " << filename << "...";
    int N = GetN();
    fwrite(&N, sizeof(int), 1, fp);
    fwrite(offset_vec_.data(), sizeof(long), offset_vec_.size(), fp);
    long eff_size = GetCacheSize();
    fwrite(data_, sizeof(char), eff_size, fp);
    fclose(fp);
    cout << "done" << endl;
}

void BaseDataCache::Print() {
    if (!is_init)
        cout << "BaseDataCache not initialized!" << endl;
    else {
        cout << "============= Info about BaseDataCache ===========\n";
        cout << "N: " << GetN() << endl;
        cout << "Size: " << GetCacheSize()
            << ", Capcaity: " << GetCacheCapacity() << endl;
        cout << "offset_vec_: ";
        for (int i = 0; i < 20 && i < GetN(); i++) {
            cout << offset_vec_[i] << " ";
        }
        cout << endl << endl;
    }
}

void ImageDataCache::Release() {
    BaseDataCache::Release();
    image_list_.clear();
    image2id_.clear();
    cache_dir_ = "";
    image_root_ = "";
}

bool ImageDataCache::AppendImageData(string image_path) {
    FILE *fp = fopen(image_path.c_str(), "rb");
    if (!fp) {
        cout << image_path << " failed" << endl;
        return false;
    }
    const int MaxFileSize = 2000000;
    unsigned char buf[MaxFileSize];  // buffer for JPEG data, 2MB
    fseek(fp, 0, SEEK_END);
    long image_size = ftell(fp);   // get the size of the JPEG data
    if (image_size > MaxFileSize) {
        cout << image_path << " file too large: " << image_size << endl;
        return false;
    }
    fseek(fp, 0, SEEK_SET); // move fp to the beginning
    fread(buf, sizeof(char), image_size, fp); // read JPEG data
    fclose(fp);

    return AppendData(buf, image_size);
}

bool ImageDataCache::LoadCache(string cache_dir, int start, int end) {
    if (!boost::filesystem::exists(cache_dir)) {
        cout << "Cache dir does not exist!" << endl;
        return false;
    }
    cache_dir_ = cache_dir;
    string idx_file = cache_dir + "/data.index";
    ifstream infile(idx_file.c_str());
    infile >> image_root_;
    int K;
    infile >> K;
    vector<long> filesize_vec(K);
    for (int i = 0; i < K; i++) {
        infile >> filesize_vec[i];
    }
    infile.close();
    if (end == 0)
        end = K;
    if (start < 0 || end > K || start >= end) {
        cout << "Invalid: " << start << ", " << end << endl;
        return false;
    }
    long total_size = 0;
    for (int i = start; i < end; i++)
        total_size += filesize_vec[i];

    Init(total_size);
    char data_file[MAX_FILENAME_LENGTH];
    char imagename_file[MAX_FILENAME_LENGTH];
    for (int i = start; i < end; i++) {
        snprintf(data_file, MAX_FILENAME_LENGTH, "%s/data.%03d",
                        cache_dir.c_str(), i);
        snprintf(imagename_file, MAX_FILENAME_LENGTH, "%s/imagename.%03d",
                        cache_dir.c_str(), i);
        cout << "Loading " << data_file << "...";
        bool ret = AppendFile(data_file);
        if (!ret) return false;
        ret = AppendImageList(imagename_file);
        if (!ret) return false;
        cout << "done" << endl;
    }
    Print();
    return true;
}

bool ImageDataCache::AppendImageList(string imagename_file) {
    ifstream infile(imagename_file.c_str());
    string tmp;
    while (infile >> tmp) {
        image2id_[tmp] = image_list_.size();
        image_list_.push_back(tmp);
    }
    infile.close();
    if (image_list_.size() != GetN()) {
        cout << imagename_file << ": number mismatch!" << endl;
        return false;
    }
    return true;
}

void ImageDataCache::WriteImageList(string imagename_file) {
    ofstream outfile(imagename_file.c_str());
    for (int i = 0; i < image_list_.size(); i++)
        outfile << image_list_[i] << endl;
    outfile.close();
}

void ImageDataCache::BuildCacheFromImageList(string image_root,
                                             string image_list_file,
                                             string cache_dir) {
    if (!boost::filesystem::exists(image_list_file)) {
        cout << "Image list does not exist!" << endl;
        return;
    }

    if (boost::filesystem::exists(cache_dir)) {
        cout << "Cache dir already exists!" << endl;
        return;
    }
    boost::filesystem::create_directory(cache_dir);

    const int batch_size = 1000;
    const int size_per_image = 400000;  //400KB
    const int cache_cap = batch_size * size_per_image;
    Release();
    Init(cache_cap);

    ifstream infile(image_list_file.c_str());
    string tmp;
    vector<string> image_list;
    while (infile >> tmp)
        image_list.push_back(tmp);
    infile.close();
    cout << image_list.size() << " images in total!" << endl;

    vector<long> filesize_vec;
    char data_file[MAX_FILENAME_LENGTH];
    char imagename_file[MAX_FILENAME_LENGTH];
    int file_idx = 0;
    for (int i = 0; i < image_list.size(); i++) {
        AppendImageData(image_root + "/" + image_list[i]);
        image_list_.push_back(image_list[i]);
        if ((i+1) % batch_size == 0) {
            snprintf(data_file, MAX_FILENAME_LENGTH, "%s/data.%03d",
                        cache_dir.c_str(), file_idx);
            snprintf(imagename_file, MAX_FILENAME_LENGTH, "%s/imagename.%03d",
                        cache_dir.c_str(), file_idx);
            file_idx++;

            WriteToFile(data_file);
            WriteImageList(imagename_file);
            filesize_vec.push_back(GetCacheSize());

            Release();
            Init(cache_cap);
        }
    }
    if (GetCacheSize() > 0) {
        snprintf(data_file, MAX_FILENAME_LENGTH, "%s/data.%03d",
                    cache_dir.c_str(), file_idx);
        snprintf(imagename_file, MAX_FILENAME_LENGTH, "%s/imagename.%03d",
                    cache_dir.c_str(), file_idx);
        WriteToFile(data_file);
        WriteImageList(imagename_file);
        filesize_vec.push_back(GetCacheSize());
        Release();
    }
    string idx_file = cache_dir + "/data.index";
    ofstream outfile(idx_file.c_str());
    outfile << image_root << endl;
    outfile << filesize_vec.size() << endl;
    for (int i = 0; i < filesize_vec.size(); i++)
        outfile << filesize_vec[i] << endl;
    outfile.close();
}

IplImage* ImageDataCache::GetImage(int i) const {
    long size = 0L;
    const char *img_data = GetData(i, size);
    if (!img_data)
        return NULL;
    CvMat mat = cvMat(1, (int)size, CV_8UC1, (char *)img_data);
    return cvDecodeImage(&mat, CV_LOAD_IMAGE_COLOR);
}

void ImageDataCache::ShowImage(int i) const {
    IplImage *img = GetImage(i);
    if (!img) {
        cout << "Get Null Image!" << endl;
        return;
    }
    cvNamedWindow("image");
    cvShowImage("image", img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvDestroyAllWindows();
}

}
