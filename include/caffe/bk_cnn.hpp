#ifndef BK_CNN_
#define BK_CNN_
#include <string>
using namespace std;


namespace BkCnn {
    enum DeviceType {CPU, GPU};

    class BkCnnAPI {
     public:
        static void InitCnnModel(string def, string model, string mean,
                        DeviceType dev_type = GPU, int dev_id = 0);

        static float GetVideoScore(const char *imagelist, float *scores);

        static void ReleaseCnnModel();
    };
}
#endif
