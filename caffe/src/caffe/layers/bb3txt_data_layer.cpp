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
#include <boost/random/uniform_real_distribution.hpp>

#include "caffe/layers/bb3txt_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// The maximum number of bounding boxes (annotations) in one image - we set the label blob shape according
// to this number
#define MAX_NUM_BBS_PER_IMAGE 20


namespace caffe {

namespace {


    /**
     * @brief Computes the number of bounding boxes in the annotation
     * @param labels Annotation of one image (dimensions MAX_NUM_BBS_PER_IMAGE x 12 x 1 x 1)
     * @param b Image id in the batch
     * @return Number of bounding boxes in this label
     */
    template <typename Dtype>
    int numBBs (const Blob<Dtype> &labels)
    {
        for (int i = 0; i < labels.shape(1); ++i)
        {
            // Data are stored like this [label, ...]
            const Dtype *data = labels.cpu_data() + labels.offset(i);
            // If the label is -1, there are no more bounding boxes
            if (data[0] == Dtype(-1.0f)) return i;
        }

        return labels.shape(1);
    }

}

template <typename Dtype>
BB3TXTDataLayer<Dtype>::BB3TXTDataLayer (const LayerParameter &param)
    : BasePrefetchingDataLayer<Dtype>(param),
      InternalThreadpool(std::max(int(boost::thread::hardware_concurrency()/2), 1))
{
}


template <typename Dtype>
BB3TXTDataLayer<Dtype>::~BB3TXTDataLayer<Dtype> ()
{
    this->StopInternalThreadpool();
    this->StopInternalThread();
}


template <typename Dtype>
void BB3TXTDataLayer<Dtype>::DataLayerSetUp (const vector<Blob<Dtype>*> &bottom,
                                            const vector<Blob<Dtype>*> &top)
{
    CHECK(this->layer_param_.has_bbtxt_param()) << "BBTXTParam is mandatory!";
    CHECK(this->layer_param_.bbtxt_param().has_height()) << "Height must be set!";
    CHECK(this->layer_param_.bbtxt_param().has_width()) << "Width must be set!";
    CHECK(this->layer_param_.bbtxt_param().has_reference_size_min()) << "Min reference size must be set!";
    CHECK(this->layer_param_.bbtxt_param().has_reference_size_max()) << "Max reference size must be set!";
    CHECK_LT(this->layer_param_.bbtxt_param().reference_size_min(),
             this->layer_param_.bbtxt_param().reference_size_max()) << "Min reference must be lower than max";

    const int height     = this->layer_param_.bbtxt_param().height();
    const int width      = this->layer_param_.bbtxt_param().width();
    const int batch_size = this->layer_param_.image_data_param().batch_size();

    this->_rng.reset(new Caffe::RNG(caffe_rng_rand()));

    // Load the BBTXT file with 2D bounding box annotations
    this->_loadBB3TXTFile();
    this->_i_global = 0;

    CHECK(!this->_images.empty()) << "The given BBTXT file is empty!";
    LOG(INFO) << "There are " << this->_images.size() << " images in the dataset.";


    // This is the shape of the input blob
    std::vector<int> top_shape = {batch_size, 3, height, width};
    this->transformed_data_.Reshape(top_shape);  // For prefetching
    top[0]->Reshape(top_shape);

    // Label blob
    std::vector<int> label_shape = {batch_size, MAX_NUM_BBS_PER_IMAGE, 12};
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
void BB3TXTDataLayer<Dtype>::load_batch (Batch<Dtype> *batch)
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
void BB3TXTDataLayer<Dtype>::InternalThreadpoolEntry (int t)
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
void BB3TXTDataLayer<Dtype>::_loadBB3TXTFile ()
{
    const std::string& source = this->layer_param_.image_data_param().source();

    std::ifstream infile(source.c_str(), std::ios::in);
    CHECK(infile.is_open()) << "BB3TXT file '" << source << "' could not be opened!";

    std::string line;
    std::vector<std::string> data;
    std::string current_filename = "";
    int i = 0;

    // Read the whole file and create entries in the _images for all images
    while (std::getline(infile, line))
    {
        // Split the line - entries separated by space:
        // [filename label confidence fblx fbly fbrx fbry rblx rbly ftly]
        boost::split(data, line, boost::is_any_of(" "));
        CHECK_EQ(data.size(), 14) << "Line '" << line << "' corrupted!";

        if (current_filename != data[0])
        {
            // This is a label to a new image
            if (this->_images.size() > 0 && i < MAX_NUM_BBS_PER_IMAGE)
            {
                // Finalize the last annotation - we put -1 as next bounding box label to signalize
                // the end - this is because each image can have a different number of bounding boxes
                int offset = this->_images.back().second->offset(i);
                Dtype* bb3_position = this->_images.back().second->mutable_cpu_data() + offset;
                bb3_position[0] = Dtype(-1.0f);
            }

            CHECK(boost::filesystem::exists(data[0])) << "File '" << data[0] << "' not found!";

            // Create new image entry
            this->_images.emplace_back(data[0], std::make_shared<Blob<Dtype>>(MAX_NUM_BBS_PER_IMAGE, 12, 1, 1));
            i = 0;
            current_filename = data[0];
        }

        // Write the bounding box info into the blob
        if (i < MAX_NUM_BBS_PER_IMAGE)
        {
            int offset = this->_images.back().second->offset(i);
            Dtype* bb3_position = this->_images.back().second->mutable_cpu_data() + offset;

            bb3_position[0] = Dtype(std::stof(data[1])); // label
            // xmin, ymin, xmax, ymax, fblx, fbly, fbrx, fbry, rblx, rbly, ftly
            for (int p = 1; p < 12; ++p) bb3_position[p] = Dtype(std::stof(data[p+2]));

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
        Dtype* bb3_position = this->_images.back().second->mutable_cpu_data() + offset;
        bb3_position[0] = Dtype(-1.0f);
    }
}


template <typename Dtype>
void BB3TXTDataLayer<Dtype>::_shuffleBoundingBoxes ()
{
    caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(_rng->generator());
    shuffle(this->_indices.begin(), this->_indices.end(), prefetch_rng);
}


template <typename Dtype>
void BB3TXTDataLayer<Dtype>::_cropAndTransform (const cv::Mat &cv_img, int b, int bb_id)
{
    CHECK_EQ(cv_img.channels(), 3) << "Image must have 3 color channels";

    // Crop this bounding box from the image
    cv::Mat cv_img_cropped;
    this->_cropBBFromImage(cv_img, cv_img_cropped, b, bb_id);

    // Mirror
    // We cannot do it here because we do not have all the needed coordinates - we have to do it externaly

    // Rotate

    // Copy the cropped and transformed image to the input blob
    this->_applyPixelTransformationsAndCopyOut(cv_img_cropped, b);
}


template <typename Dtype>
void BB3TXTDataLayer<Dtype>::_cropBBFromImage (const cv::Mat &cv_img, cv::Mat &cv_img_cropped_out,
                                              int b, int bb_id)
{
    // Input dimensions of the network
    const int height             = this->layer_param_.bbtxt_param().height();
    const int width              = this->layer_param_.bbtxt_param().width();
    const int reference_size_min = this->layer_param_.bbtxt_param().reference_size_min();
    const int reference_size_max = this->layer_param_.bbtxt_param().reference_size_max();
    caffe::rng_t* rng            = static_cast<caffe::rng_t*>(this->_rng->generator());


    // Get dimensions of the bounding box - format [label, xmin, ymin, xmax, ymax]
    const Dtype *bb_data = this->transformed_label_.cpu_data() + this->transformed_label_.offset(b, bb_id);
    const Dtype x = bb_data[1];
    const Dtype y = bb_data[2];
    const Dtype w = bb_data[3] - bb_data[1];
    const Dtype h = bb_data[4] - bb_data[2];

    int reference_size;
    if (this->phase_ == TRAIN)
    {
        // Choose the reference bounding box size within the given range

        // IMPORTANT! We need larger values to be less probable than small - because an accumulator, which
        // detects larger objects spans across more "integer" sizes. For example accumulator x2 can detect
        // bbs in [25,60], then x4 detects bbs from [50,120], which has double the size of the interval. The
        // random distribution needs to be augmented accordingly - we will project the bb sizes to log space
        double rsl_min = std::log(double(reference_size_min));
        double rsl_max = std::log(double(reference_size_max));

        boost::random::uniform_real_distribution<double> dists(0.0, 1.0);
        reference_size = std::round(std::exp(rsl_min + (dists(*rng) * (rsl_max-rsl_min))));
    }
    else
    {
        // Testing phase - keep the original size
        reference_size = std::max(w, h);

        if (reference_size < reference_size_min) reference_size = reference_size_min;
        if (reference_size > reference_size_max) reference_size = reference_size_max;
    }

    const Dtype size      = std::max(w, h);
    const int crop_width  = std::ceil(double(width) / reference_size * size);
    const int crop_height = std::ceil(double(height) / reference_size * size);

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


    // -- CROP THE IMAGE -- //
    // Because we want to save memory we will not use the copyMakeBorder function to create the border, but
    // instead we will compute an intersection of the crop with the image, crop that intersection, scale it
    // and then place it on black background to the position in which it would be cropped from

    // Intersection of the crop with the image
    int ints_width  = crop_width - border_left - border_right;
    int ints_height = crop_height - border_top - border_bottom;
    cv::Rect intersection(crop_x+border_left, crop_y+border_top, ints_width, ints_height);

    // Scale the intersection according to the crip scaling ratio (do not round here because the crop could
    // escape from the network input size - conversion to int will floor it)
    int ints_width_scaled  = ints_width / size * reference_size;
    int ints_height_scaled = ints_height / size * reference_size;
    int border_left_scaled = border_left / size * reference_size;  // x coordinate of the crop
    int border_top_scaled  = border_top / size * reference_size;   // y coordinate of the crop
    // Corrections for imprecisions - we have to keep the crop inside of the network input image dimensions
    if (ints_width_scaled > width) ints_width_scaled = width;
    if (ints_height_scaled > height) ints_height_scaled = height;
    if (border_left_scaled+ints_width_scaled > width) border_left_scaled = width - ints_width_scaled;
    if (border_top_scaled+ints_height_scaled > height) border_top_scaled = height - ints_height_scaled;

    CHECK_GT(ints_width_scaled, 0) << "Crop does not intersect the image";
    CHECK_GT(ints_height_scaled, 0) << "Crop does not intersect the image";
    CHECK_LE(ints_width_scaled, width) << "Crop larger than width: " << ints_width_scaled;
    CHECK_LE(ints_height_scaled, height) << "Crop larger than height: " << ints_height_scaled;
    CHECK_LE(border_left_scaled+ints_width_scaled, width) << "Moved crop does not fit in the image";
    CHECK_LE(border_top_scaled+ints_height_scaled, height) << "Moved crop does not fit in the image";

    // Crop and scale down the cropped intersection
    cv::Mat cv_img_cropped_scaled;
    cv::resize(cv_img(intersection), cv_img_cropped_scaled, cv::Size(ints_width_scaled, ints_height_scaled));

    // Initialize the network input with black
    cv_img_cropped_out = cv::Mat::zeros(height, width, CV_8UC3);

    // Place the crop onto black canvas of the size of the network input image
    cv::Mat cv_img_subcrop = cv_img_cropped_out(cv::Rect(border_left_scaled, border_top_scaled,
                                                         ints_width_scaled, ints_height_scaled));
    cv_img_cropped_scaled.copyTo(cv_img_subcrop);


    // -- UPDATE THE COORDINATES OF THE BOUNDING BOXES -- //
    // Update the bounding box coordinates - we need to update all annotations
    Dtype x_scaling   = float(width) / crop_width;
    Dtype y_scaling   = float(height) / crop_height;
    for (int i = 0; i < MAX_NUM_BBS_PER_IMAGE; ++i)
    {
        // Data are stored like this [label, xmin, ymin, xmax, ymax, fblx, fbly, fbrx, fbry, rblx, rbly, ftly]
        Dtype *data = this->transformed_label_.mutable_cpu_data() + this->transformed_label_.offset(b, i);

        if (data[0] == Dtype(-1.0f)) break;

        // Align with x, y of the crop and scale
        data[1]  = (data[1]-crop_x) * x_scaling; // xmin
        data[2]  = (data[2]-crop_y) * y_scaling; // ymin
        data[3]  = (data[3]-crop_x) * x_scaling; // xmax
        data[4]  = (data[4]-crop_y) * y_scaling; // ymax
        data[5]  = (data[5]-crop_x) * x_scaling; // fblx
        data[6]  = (data[6]-crop_y) * y_scaling; // fbly
        data[7]  = (data[7]-crop_x) * x_scaling; // fbrx
        data[8]  = (data[8]-crop_y) * y_scaling; // fbry
        data[9]  = (data[9]-crop_x) * x_scaling; // rblx
        data[10] = (data[10]-crop_y) * y_scaling; // rbly
        data[11] = (data[11]-crop_y) * y_scaling; // ftly

//        cv::rectangle(cv_img_cropped_out, cv::Rect(data[1], data[2], data[3]-data[1], data[4]-data[2]), cv::Scalar(0,0,255), 2);
    }
//    static int imi = 0;
//    if(this->phase_ == TRAIN) cv::imwrite("cropped" + std::to_string(imi++) + ".png", cv_img_cropped_out);
}


template <typename Dtype>
void BB3TXTDataLayer<Dtype>::_applyPixelTransformationsAndCopyOut (const cv::Mat &cv_img_cropped, int b)
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

        boost::random::uniform_int_distribution<> diste(-40, 50);  // Exposure
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
SelectedBB<Dtype> BB3TXTDataLayer<Dtype>::_getImageAndBB()
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

INSTANTIATE_CLASS(BB3TXTDataLayer);
REGISTER_LAYER_CLASS(BB3TXTData);


}  // namespace caffe
#endif  // USE_OPENCV
