#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/bbtxt_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// The maximum number of bounding boxes (annotations) in one image - we set the label blob shape according
// to this number
#define MAX_NUM_BBS_PER_IMAGE 20


namespace caffe {

template <typename Dtype>
BBTXTDataLayer<Dtype>::BBTXTDataLayer (const LayerParameter &param)
    : BasePrefetchingDataLayer<Dtype>(param)
{
}


template <typename Dtype>
BBTXTDataLayer<Dtype>::~BBTXTDataLayer<Dtype> ()
{
    this->StopInternalThread();
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::DataLayerSetUp (const vector<Blob<Dtype>*> &bottom,
                                            const vector<Blob<Dtype>*> &top)
{
    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width  = this->layer_param_.image_data_param().new_width();
    const bool is_color  = this->layer_param_.image_data_param().is_color();

    CHECK((new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0)) << "Current "
        "implementation requires new_height and new_width to be set at the same time.";


    // Load the BBTXT file with 2D bounding box annotations
    this->_loadBBTXTFile();
    this->_i_global = 0;

    CHECK(!this->_images.empty()) << "The given BBTXT file is empty!";
    LOG(INFO) << "There are " << this->_images.size() << " images in the dataset set.";

    if (this->layer_param_.image_data_param().shuffle())
    {
        // Initialize the random number generator for shuffling and shuffle the images
        this->_rng.reset(new Caffe::RNG(caffe_rng_rand()));
        this->_shuffleImages();
    }


    // Read an image, and use it to initialize the top blob
    cv::Mat cv_img = ReadImageToCVMat(this->_images[this->_i_global].first, new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not open " << this->_images[this->_i_global].first;

    // Use data_transformer to infer the expected blob shape from a cv_image
    std::vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);

    // We need this to speed up batch loading because we do not want to allocate new memory every time
    this->transformed_data_.Reshape(top_shape);

    // Reshape prefetch_data and top[0] according to the batch_size
    int batch_size = this->layer_param_.image_data_param().batch_size();
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);

    LOG(INFO) << "BBTXTDataLayer bottom shape: " << top[0]->shape(0) << " x " << top[0]->shape(1) << " x "
              << top[0]->shape(2) << " x " << top[0]->shape(3);


    // Initialize prefetching
    // We also have to reshape the prefetching blobs to the correct batch size
    for (int i = 0; i < this->prefetch_.size(); ++i) this->prefetch_[i]->data_.Reshape(top_shape);
    // Label blob
    std::vector<int> label_shape = {batch_size, MAX_NUM_BBS_PER_IMAGE, 5};
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) this->prefetch_[i]->label_.Reshape(label_shape);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::load_batch (Batch<Dtype> *batch)
{
    // This function is called on prefetch thread

    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width  = this->layer_param_.image_data_param().new_width();
    const bool is_color  = this->layer_param_.image_data_param().is_color();



    Dtype* prefetch_data  = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();


    for (int b = 0; b < batch_size; ++b)
    {
        // get a blob
        timer.Start();
        cv::Mat cv_img = ReadImageToCVMat(this->_images[this->_i_global].first, new_height,
                                          new_width, is_color);
        CHECK(cv_img.data) << "Could not open " << this->_images[this->_i_global].first;
        read_time += timer.MicroSeconds();

        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset_image = batch->data_.offset(b);
        this->transformed_data_.set_cpu_data(prefetch_data + offset_image);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        trans_time += timer.MicroSeconds();

        // Copy the annotation
        int offset_label = batch->label_.offset(b);
        std::shared_ptr<Blob<Dtype>> plabel = this->_images[this->_i_global].second;
        caffe_copy(plabel->count(), plabel->cpu_data(), prefetch_label+offset_label);

        // Move index to the next image
        this->_i_global++;
        if (this->_i_global >= this->_images.size())
        {
            // Restart the counter from the begining
            if (this->layer_param_.image_data_param().shuffle()) this->_shuffleImages();
            this->_i_global = 0;
        }
    }

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}



// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

template <typename Dtype>
void BBTXTDataLayer<Dtype>::_loadBBTXTFile ()
{
    const std::string& source = this->layer_param_.image_data_param().source();

    std::ifstream infile(source.c_str(), std::ios::in);
    CHECK(infile.is_open()) << "BBTXT file '" << source << "' could not be opened!";

    std::string line;
    std::vector<std::string> data;
    std::string current_filename = "";
    int i = 0;

    // Read the whole file and create entries in the _images for all images
    while (std::getline(infile, line))
    {
        // Split the line - entries separated by space [filename label confidence xmin ymin xmax ymax]
        boost::split(data, line, boost::is_any_of(" "));
        CHECK_EQ(data.size(), 7) << "Line '" << line << "' corrupted!";

        if (current_filename != data[0])
        {
            // This is a label to a new image
            if (this->_images.size() > 0 && i < MAX_NUM_BBS_PER_IMAGE)
            {
                // Finalize the last annotation - we put 0s as next bounding box to signalize the end - this
                // is because each image can have a different number of bounding boxes
                int offset = this->_images.back().second->offset(i);
                Dtype* bb_position = this->_images.back().second->mutable_cpu_data() + offset;
                for (int j = 0; j < 5; j++) bb_position[j] = Dtype(0.0f);
            }

            CHECK(boost::filesystem::exists(data[0])) << "File '" << data[0] << "' not found!";

            // Create new image entry
            this->_images.push_back(std::make_pair(data[0],
                                        std::make_shared<Blob<Dtype>>(MAX_NUM_BBS_PER_IMAGE, 5, 1, 1)));
            i = 0;
            current_filename = data[0];
        }

        // Write the bounding box info into the blob
        if (i < MAX_NUM_BBS_PER_IMAGE)
        {
            int offset = this->_images.back().second->offset(i);
            Dtype* bb_position = this->_images.back().second->mutable_cpu_data() + offset;
            bb_position[0] = Dtype(std::stof(data[1])); // label
            bb_position[1] = Dtype(std::stof(data[3])); // xmin
            bb_position[2] = Dtype(std::stof(data[4])); // ymin
            bb_position[3] = Dtype(std::stof(data[5])); // xmax
            bb_position[4] = Dtype(std::stof(data[6])); // ymax
            i++;
        }
        else
        {
            LOG(WARNING) << "Skipping bb - max number of bounding boxes per image reached.";
        }
    }
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_shuffleImages ()
{
    caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(_rng->generator());
    shuffle(this->_images.begin(), this->_images.end(), prefetch_rng);
}



// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

INSTANTIATE_CLASS(BBTXTDataLayer);
REGISTER_LAYER_CLASS(BBTXTData);


}  // namespace caffe
#endif  // USE_OPENCV
