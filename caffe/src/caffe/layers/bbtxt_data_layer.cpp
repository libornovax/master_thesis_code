#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/layers/bbtxt_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/internal_threadpool.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_counter.hpp"

// The maximum number of bounding boxes (annotations) in one image - we set the label blob shape according
// to this number
#define MAX_NUM_BBS_PER_IMAGE 20


namespace caffe {

namespace {


    /**
     * @brief Computes the number of bounding boxes in the annotation
     * @param labels Annotation of one image (dimensions MAX_NUM_BBS_PER_IMAGE x 5 x 1 x 1)
     * @param b Image id in the batch
     * @return Number of bounding boxes in this label
     */
    template <typename Dtype>
    int numBBs (const Blob<Dtype> &labels)
    {
        for (int i = 0; i < labels.shape(1); ++i)
        {
            // Data are stored like this [label, xmin, ymin, xmax, ymax]
            const Dtype *data = labels.cpu_data() + labels.offset(i);
            // If the label is -1, there are no more bounding boxes
            if (data[0] == Dtype(-1.0f)) return i;
        }

        return labels.shape(1);
    }

}

template <typename Dtype>
BBTXTDataLayer<Dtype>::BBTXTDataLayer (const LayerParameter &param)
    : BasePrefetchingDataLayer<Dtype>(param),
      InternalThreadpool(std::max(int(boost::thread::hardware_concurrency()/2), 1))
{
}


template <typename Dtype>
BBTXTDataLayer<Dtype>::~BBTXTDataLayer<Dtype> ()
{
    this->StopInternalThreadpool();
    this->StopInternalThread();
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::DataLayerSetUp (const vector<Blob<Dtype>*> &bottom,
                                            const vector<Blob<Dtype>*> &top)
{
    CHECK(this->layer_param_.has_bbtxt_param()) << "BBTXTParam is mandatory!";
    CHECK(this->layer_param_.bbtxt_param().has_height()) << "Height must be set!";
    CHECK(this->layer_param_.bbtxt_param().has_width()) << "Width must be set!";
    CHECK(this->layer_param_.bbtxt_param().has_reference_size()) << "Reference size must be set!";

    const int height     = this->layer_param_.bbtxt_param().height();
    const int width      = this->layer_param_.bbtxt_param().width();
    const int batch_size = this->layer_param_.image_data_param().batch_size();

    this->_rng.reset(new Caffe::RNG(caffe_rng_rand()));

    // Load the BBTXT file with 2D bounding box annotations
    this->_loadBBTXTFile();
    this->_i_global = 0;

    CHECK(!this->_images.empty()) << "The given BBTXT file is empty!";
    LOG(INFO) << "There are " << this->_images.size() << " images in the dataset.";

    if (this->phase_ == TRAIN)
    {
        // Initialize the random number generator for shuffling and shuffle the images
        this->_shuffleBoundingBoxes();
    }


    // This is the shape of the input blob
    std::vector<int> top_shape = {batch_size, 3, height, width};
    this->transformed_data_.Reshape(top_shape);  // For prefetching
    top[0]->Reshape(top_shape);

    // Label blob
    std::vector<int> label_shape = {batch_size, MAX_NUM_BBS_PER_IMAGE, 5};
    this->transformed_label_.Reshape(label_shape);  // For prefetching
    top[1]->Reshape(label_shape);


    // Initialize prefetching
    // We also have to reshape the prefetching blobs to the correct batch size
    for (int i = 0; i < this->prefetch_.size(); ++i)
    {
        this->prefetch_[i]->data_.Reshape(top_shape);
        this->prefetch_[i]->label_.Reshape(label_shape);
    }

    this->StartInternalThreadpool();
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::load_batch (Batch<Dtype> *batch)
{
    // This function is called on a prefetch thread

    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    const int batch_size = this->layer_param_.image_data_param().batch_size();

    Dtype* prefetch_data  = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    this->transformed_data_.set_cpu_data(prefetch_data);
    this->transformed_label_.set_cpu_data(prefetch_label);

    this->_num_processed.reset();
    for (int b = 0; b < batch_size; ++b) this->_b_queue.push(b);
    this->_num_processed.waitToCount(batch_size);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::InternalThreadpoolEntry (int t)
{
    // This method runs on each thread of the internal threadpool

    std::cout << "============================= STARTING DATA THREAD " << t << std::endl;

    try {
        while (!this->must_stopt(t))
        {
            int b = this->_b_queue.pop();

            // Get index of image and bounding box we will crop
            SelectedBB<Dtype> selbb = this->_getImageAndBB();

            cv::Mat cv_img = cv::imread(selbb.filename, CV_LOAD_IMAGE_COLOR);
            CHECK(cv_img.data) << "Could not open " << selbb.filename;

            // Copy the annotation - we really have to copy it because it will be altered during image
            // transformations like cropping or scaling
            caffe_copy(selbb.label->count(), selbb.label->cpu_data(),
                       this->transformed_label_.mutable_cpu_data() + this->transformed_label_.offset(b));

            // We select a bounding box from the image and then make a crop such that the bounding box is
            // inside of it and it has the reference size (Training - we select a random bounding box to crop
            // from each image, Testing - crop all bounding boxes from the image - this way we ensure
            // the test set is always the same)
            this->_cropAndTransform(cv_img, b, selbb.bb_id);

            // Raise the counter on processed images
            this->_num_processed.increase();

        }
    } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
    }

    std::cout << "============================= KILLING DATA THREAD " << t << std::endl;
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
                // Finalize the last annotation - we put -1 as next bounding box label to signalize
                // the end - this is because each image can have a different number of bounding boxes
                int offset = this->_images.back().second->offset(i);
                Dtype* bb_position = this->_images.back().second->mutable_cpu_data() + offset;
                bb_position[0] = Dtype(-1.0f);
            }

            CHECK(boost::filesystem::exists(data[0])) << "File '" << data[0] << "' not found!";

            // Create new image entry
            this->_images.emplace_back(data[0], std::make_shared<Blob<Dtype>>(MAX_NUM_BBS_PER_IMAGE, 5, 1, 1));
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

            this->_indices.emplace_back(this->_images.size()-1, i);
            i++;
        }
        else
        {
            LOG(WARNING) << "Skipping bb - max number of bounding boxes per image reached.";
        }
    }

    // Close the last annotation
    if (i < MAX_NUM_BBS_PER_IMAGE)
    {
        int offset = this->_images.back().second->offset(i);
        Dtype* bb_position = this->_images.back().second->mutable_cpu_data() + offset;
        bb_position[0] = Dtype(-1.0f);
    }
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_shuffleBoundingBoxes ()
{
    caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(_rng->generator());
    shuffle(this->_indices.begin(), this->_indices.end(), prefetch_rng);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_cropAndTransform (const cv::Mat &cv_img, int b, int bb_id)
{
    CHECK_EQ(cv_img.channels(), 3) << "Image must have 3 color channels";

    // Crop this bounding box from the image
    cv::Mat cv_img_cropped;
    this->_cropBBFromImage(cv_img, cv_img_cropped, b, bb_id);

    // Mirror
    this->_flipImage(cv_img_cropped, b);

    // Rotate

    // Copy the cropped and transformed image to the input blob
    this->_applyPixelTransformationsAndCopyOut(cv_img_cropped, b);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_cropBBFromImage (const cv::Mat &cv_img, cv::Mat &cv_img_cropped_out,
                                              int b, int bb_id)
{
    // Input dimensions of the network
    const int height   = this->layer_param_.bbtxt_param().height();
    const int width    = this->layer_param_.bbtxt_param().width();
    // This is the size of the bounding box that is detected by this network
    int reference_size = this->layer_param_.bbtxt_param().reference_size();
    caffe::rng_t* rng  = static_cast<caffe::rng_t*>(this->_rng->generator());

    if (this->phase_ == TRAIN)
    {
        // Vary the reference bounding box size a bit, i.e. the size of the object when it is cropped
        boost::random::uniform_int_distribution<> dists(-reference_size/10, reference_size/5);
        reference_size += dists(*rng);
    }


    // Get dimensions of the bounding box - format [label, xmin, ymin, xmax, ymax]
    const Dtype *bb_data = this->transformed_label_.cpu_data() + this->transformed_label_.offset(b, bb_id);
    const Dtype x = bb_data[1];
    const Dtype y = bb_data[2];
    const Dtype w = bb_data[3] - bb_data[1];
    const Dtype h = bb_data[4] - bb_data[2];

    const Dtype size      = std::max(w, h);
    const int crop_width  = std::round(double(width) / reference_size * size);
    const int crop_height = std::round(double(height) / reference_size * size);

    // Select a random position of the crop, but it has to fully contain the bounding box
    int crop_x, crop_y;
    if (this->phase_ == TRAIN)
    {
        // In training we want to create a random crop around the bounding box
        boost::random::uniform_int_distribution<> distx(x+w-crop_width, x);
        boost::random::uniform_int_distribution<> disty(y+h-crop_height, y);
        crop_x = distx(*rng);
        crop_y = disty(*rng);
    }
    else
    {
        // In testing we want to always crop the same crop. Thus we will place the bounding box to the center
        // of the crop
        crop_x = x + w/2 - crop_width/2;
        crop_y = y + h/2 - crop_height/2;
    }

    // Now if the crop spans outside the image we have to pad the image
    int border_left   = (crop_x < 0) ? -crop_x : 0;
    int border_top    = (crop_y < 0) ? -crop_y : 0;
    int border_right  = (crop_x+crop_width  > cv_img.cols) ? (crop_x+crop_width  - cv_img.cols) : 0;
    int border_bottom = (crop_y+crop_height > cv_img.rows) ? (crop_y+crop_height - cv_img.rows) : 0;

    cv::Mat cv_img_padded;
    cv::copyMakeBorder(cv_img, cv_img_padded, border_top, border_bottom, border_left, border_right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    // Crop
    cv_img_cropped_out = cv_img_padded(cv::Rect(crop_x+border_left, crop_y+border_top, crop_width, crop_height));
    cv::resize(cv_img_cropped_out, cv_img_cropped_out, cv::Size(width, height));


    // Update the bounding box coordinates - we need to update all annotations
    Dtype x_scaling   = float(width) / crop_width;
    Dtype y_scaling   = float(height) / crop_height;
    for (int i = 0; i < MAX_NUM_BBS_PER_IMAGE; ++i)
    {
        // Data are stored like this [label, xmin, ymin, xmax, ymax]
        Dtype *data = this->transformed_label_.mutable_cpu_data() + this->transformed_label_.offset(b, i);

        if (data[0] == Dtype(-1.0f)) break;

        // Align with x, y of the crop and scale
        data[1] = (data[1]-crop_x) * x_scaling;
        data[2] = (data[2]-crop_y) * y_scaling;
        data[3] = (data[3]-crop_x) * x_scaling;
        data[4] = (data[4]-crop_y) * y_scaling;

//        cv::rectangle(cv_img_cropped_out, cv::Rect(data[1], data[2], data[3]-data[1], data[4]-data[2]), cv::Scalar(0,0,255), 2);
    }
//    static int imi = 0;
//    if(this->phase_ == TRAIN) cv::imwrite("cropped" + std::to_string(imi++) + ".png", cv_img_cropped_out);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_applyPixelTransformationsAndCopyOut (const cv::Mat &cv_img_cropped, int b)
{
    const int width  = cv_img_cropped.cols;
    const int height = cv_img_cropped.rows;

    CHECK(cv_img_cropped.data) << "Something went wrong with cropping!";
    CHECK_EQ(height, this->transformed_data_.shape(2)) << "Wrong crop height! Does not match network!";
    CHECK_EQ(width,  this->transformed_data_.shape(3)) << "Wrong crop width! Does not match network!";


    // Distortions of the image color space - during testing those will be 0
    Dtype exposure         = Dtype(0.0f);
    std::vector<Dtype> bgr = {Dtype(0.0f), Dtype(0.0f), Dtype(0.0f)};
    cv::Mat noise          = cv::Mat::zeros(height, width, CV_32FC3);
    if (this->phase_ == TRAIN)
    {
        // Apply random transformations
        caffe::rng_t* rng = static_cast<caffe::rng_t*>(this->_rng->generator());

        boost::random::uniform_int_distribution<> diste(-30, 30);  // Exposure
        exposure = diste(*rng);

        boost::random::uniform_int_distribution<> distbgr(-20, 20);  // Hue
        bgr[0] = distbgr(*rng);
        bgr[1] = distbgr(*rng);
        bgr[2] = distbgr(*rng);

        boost::random::uniform_int_distribution<> distn(0, 25);  // Noise standard deviation
        cv::randn(noise, 0, distn(*rng));
    }

    // Normalize to 0 mean and unit variance and copy the image to the transformed_image
    // + apply exposure, noise, hue, saturation, ...
    Dtype* transformed_data = this->transformed_data_.mutable_cpu_data() + this->transformed_data_.offset(b);
    for (int i = 0; i < height; ++i)
    {
        const uchar* ptr  = cv_img_cropped.ptr<uchar>(i);
        const float* ptrn = noise.ptr<float>(i);
        int img_index = 0;  // Index in the cv_img_cropped
        for (int j = 0; j < width; ++j)
        {
            for (int c = 0; c < 3; ++c)
            {
                const int top_index = (c * height + i) * width + j;
                // Apply exposure change, hue distortion, noise
                const Dtype val = Dtype(ptr[img_index]) + exposure + bgr[c] + Dtype(ptrn[img_index]);
                // Zero mean and unit variance
                transformed_data[top_index] = (std::max(Dtype(0.0f), std::min(Dtype(255.0f), val))
                        - Dtype(128.0f)) / Dtype(128.0f);
                img_index++;
            }
        }
    }

//    std::vector<cv::Mat> chnls;
//    chnls.push_back(cv::Mat(height, width, CV_32FC1, transformed_image.mutable_cpu_data()+transformed_image.offset(0,0)));
//    chnls.push_back(cv::Mat(height, width, CV_32FC1, transformed_image.mutable_cpu_data()+transformed_image.offset(0,1)));
//    chnls.push_back(cv::Mat(height, width, CV_32FC1, transformed_image.mutable_cpu_data()+transformed_image.offset(0,2)));
//    cv::Mat img; cv::merge(chnls, img);
//    img *= 128.0f;
//    img += cv::Scalar(128.0f,128.0f,128.0f);
//    cv::Mat img_u; img.convertTo(img_u, CV_8UC3);
//    static int imi = 0;
//    cv::imwrite("transfromed" + std::to_string(imi++) + ".png", img_u);
}


template <typename Dtype>
void BBTXTDataLayer<Dtype>::_flipImage(cv::Mat &cv_img_cropped, int b)
{
    if (this->phase_ == TEST) return;

    caffe::rng_t* rng  = static_cast<caffe::rng_t*>(this->_rng->generator());
    boost::random::uniform_int_distribution<> dist(0, 1);

    if (dist(*rng) == 1)
    {
        // Flip the image
        cv::flip(cv_img_cropped, cv_img_cropped, 1);

        // Flip the annotations
        const Dtype width = Dtype(cv_img_cropped.cols);
        for (int i = 0; i < MAX_NUM_BBS_PER_IMAGE; ++i)
        {
            // Data are stored like this [label, xmin, ymin, xmax, ymax]
            Dtype *data = this->transformed_label_.mutable_cpu_data() + this->transformed_label_.offset(b, i);

            if (data[0] == Dtype(-1.0f)) break;

            Dtype temp = data[1];
            data[1]  = width - data[3]; // xmax -> xmin
            data[3]  = width - temp;    // xmin -> xmax
        }
    }
}


template <typename Dtype>
SelectedBB<Dtype> BBTXTDataLayer<Dtype>::_getImageAndBB ()
{
    std::lock_guard<std::mutex> lock(this->_i_global_mtx);

    // Get image and bounding box index
    auto indices = this->_indices[this->_i_global++];

    if (this->_i_global >= this->_indices.size())
    {
        this->_i_global = 0;  // Restart
        if (this->phase_ == TRAIN) this->_shuffleBoundingBoxes();
    }

    SelectedBB<Dtype> sel;
    sel.filename = this->_images[indices.first].first;
    sel.label    = this->_images[indices.first].second;
    sel.bb_id    = indices.second;

    return sel;
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

INSTANTIATE_CLASS(BBTXTDataLayer);
REGISTER_LAYER_CLASS(BBTXTData);


}  // namespace caffe
#endif  // USE_OPENCV
