//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <string>
#include <vector>
#include <iostream>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  if (!mxIsCell(bottom))
    mexErrMsgTxt("Wrong Input Type, cell array expected!");
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) != input_blobs.size())
    mexErrMsgTxt("Input data dimension doesn't match with that of the network");
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem))
        mexErrMsgTxt("MatCaffe require single-precision float point data");
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count())
        mexErrMsgTxt("MatCaffe input size does not match the input size of the network");
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);

    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* get_blob_data(const mxArray* const b_name) {
  if(!mxIsChar(b_name)) {
      mexErrMsgTxt("get_blob require a string (blob name) as input");
  }
          // << " ";
  char* blob_name = mxArrayToString(b_name);
  if (!net_->has_blob(blob_name)) {
      mexErrMsgTxt("Cannot find layer");
      return NULL;
  }
  const shared_ptr<Blob<float> > data_blob
        = net_->blob_by_name(blob_name);
  mwSize dims[4] = {data_blob->width(), data_blob->height(),
                    data_blob->channels(), data_blob->num()};
  mxArray* mx_data =
    mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_data));
  switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(data_blob->count(), data_blob->cpu_data(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(data_blob->count(), data_blob->gpu_data(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return mx_data;
}

static mxArray* get_input_diff() {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  return mx_out;
}

static mxArray* do_backward_from(const mxArray* const l_name, const mxArray* const b_name, const mxArray* const diff) {
  if(!mxIsChar(b_name) || !mxIsChar(l_name) || !mxIsSingle(diff)) {
      mexErrMsgTxt("require a string (blob name|layer name) as input");
  }
  char *layer_name = mxArrayToString(l_name);
  if (!net_->has_layer(layer_name)) {
      mexErrMsgTxt("Cannot find layer");
      return NULL;
  }
  char *blob_name = mxArrayToString(b_name);
  if (!net_->has_blob(blob_name)) {
      mexErrMsgTxt("Cannot find blob");
      return NULL;
  }

  // Step 1. set the blob diff
  const shared_ptr<Blob<float> > diff_blob
        = net_->blob_by_name(blob_name);
  if (!mxIsSingle(diff))
    mexErrMsgTxt("Wrong input type, single expected");
  if (mxGetNumberOfElements(diff) != diff_blob->count()) {
    mexErrMsgTxt("input number of elements doesn't match the size of the network");
    return NULL;
  }
  float* diff_ptr = reinterpret_cast<float*>(mxGetPr(diff));
  switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(diff_blob->count(), diff_ptr, diff_blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(diff_blob->count(), diff_ptr, diff_blob->mutable_gpu_diff());
  }

  // Step 2. do bp from the specified layer
  net_->BackwardBypassNorm(layer_name);

  // Step 3. get the input diff
  return get_input_diff();
}


static mxArray* do_backward(const mxArray* const top_diff, const mxArray* const bp_t_) {
  if (!mxIsNumeric(bp_t_) || !mxIsCell(top_diff))
    mexErrMsgTxt("Wrong Input Type!");
  int bp_type = static_cast<int>(mxGetScalar(bp_t_));
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) != output_blobs.size())
    mexErrMsgTxt("Input data dimension doesn't match with that of the network");
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    if (!mxIsSingle(elem))
        mexErrMsgTxt("MatCaffe require single-precision float point data");
    if (mxGetNumberOfElements(elem) != output_blobs[i]->count())
        mexErrMsgTxt("input number of elements doesn't match the size of the network");

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // const float* p = output_blobs[0]->cpu_diff();
  // for (int i = 0; i < 10; i++)
  //   std::cout << *p++ << " ";
  // std::cout << std::endl;
  // LOG(INFO) << "Start";
  switch (bp_type) {
    case 0:
      net_->BackwardBypassNorm();
      // net_->Backward();
      break;
    case 1:
      std::cout << "use dummy backward" << std::endl;
      net_->DummyBackward();
      break;
    default:
      LOG(FATAL) << "Unknown bp type.";
  }
  // LOG(INFO) << "End";
  return get_input_diff();
}

static mxArray* do_backward(const mxArray* const top_diff) {
  return do_backward(top_diff, mxCreateDoubleScalar(0));
}

static mxArray* do_get_weight(const mxArray* const l_name) {
  if (!mxIsChar(l_name))
    mexErrMsgTxt("Wrong Input Type: string expected");
  char *layer_name = mxArrayToString(l_name);
  if (!net_->has_layer(layer_name))
    mexErrMsgTxt("Cannot find layer");
  shared_ptr<Layer<float> > layer_ptr = net_->layer_by_name(layer_name);
  if (layer_ptr->blobs().size() == 0)
    mexErrMsgTxt("The specified layer has no weights");
  shared_ptr<Blob<float> > blob_ptr = layer_ptr->blobs()[0]; // get weights only
  mwSize dims[4] = {blob_ptr->width(), blob_ptr->height(),
                    blob_ptr->channels(), blob_ptr->num()};
  mxArray* mx_weights =
    mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_copy(blob_ptr->count(), blob_ptr->cpu_data(), weights_ptr);
    break;
  case Caffe::GPU:
    caffe_copy(blob_ptr->count(), blob_ptr->gpu_data(), weights_ptr);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return mx_weights;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {\
  if (nrhs == 0)
    plhs[0] = do_get_weights();
  else if (nrhs == 1)
    plhs[0] = do_get_weight(prhs[0]);
  else
    mexErrMsgTxt("Wrong number of arguments");
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_forward(prhs[0]);
}

static void backward(MEX_ARGS) {
  if (nrhs == 1) {
    plhs[0] = do_backward(prhs[0]);
  }
  else if (nrhs == 2) {
    plhs[0] = do_backward(prhs[0], prhs[1]);
  }
  else if (nrhs == 3) {
    plhs[0] = do_backward_from(prhs[0], prhs[1], prhs[2]);
  }
  else {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
}

static void get_blob(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = get_blob_data(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "backward",           backward        },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  { "get_blob",           get_blob        },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
