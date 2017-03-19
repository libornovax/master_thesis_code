#include <vector>
#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/bb3txt_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/internal_threadpool.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_counter.hpp"
#include "caffe/util/benchmark.hpp"


namespace caffe {


template <typename Dtype>
BB3TXTLossLayer<Dtype>::BB3TXTLossLayer (const LayerParameter &param)
    : LossLayer<Dtype>(param),
      InternalThreadpool(4),
      _b_queue()
{
    CHECK(param.has_accumulator_loss_param()) << "AccumulatorLossParameter is mandatory!";
    CHECK(param.accumulator_loss_param().has_radius()) << "Radius is mandatory!";
    CHECK(param.accumulator_loss_param().has_circle_ratio()) << "Circle ratio is mandatory!";
    CHECK(param.accumulator_loss_param().has_downsampling()) << "Downsampling is mandatory!";
}


template <typename Dtype>
BB3TXTLossLayer<Dtype>::~BB3TXTLossLayer ()
{
    StopInternalThreadpool();
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first bottom[0] and then the accumulator bottom[1]

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    this->_accumulator = std::make_shared<Blob<Dtype>>();
    this->_diff        = std::make_shared<Blob<Dtype>>();

    // Scale of this accumulator is given by downsampling
    this->_scale = this->layer_param_.accumulator_loss_param().downsampling();
    // Compute the bounds for bounding boxes, which should be included in this accumulator
    this->_computeSizeBounds();

    this->StartInternalThreadpool();
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::Reshape (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first bottom[0] and then the accumulator bottom[1]

    LossLayer<Dtype>::Reshape(bottom, top);

    CHECK_EQ(bottom[1]->shape(1), 8) << "Accumulator must have 8 channels (prob, fblx, fbly, fbrx, fbry, "
                                     << "rblx, rbly, ftly)!";

    this->_accumulator->ReshapeLike(*bottom[1]);
    this->_diff->ReshapeLike(*bottom[1]);
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    const int batch_size = bottom[1]->shape(0);

    this->_loss_prob  = Dtype(0.0f);
    this->_loss_coord = Dtype(0.0f);

    this->_num_processed.reset();

    // We have to store the labels and net output in internal members because we will be accessing them in
    // threads
    this->_labels = bottom[0]; this->_labels->cpu_data();
    this->_bottom = bottom[1]; this->_bottom->cpu_data();

    // -- COMPUTE THE LOSS -- //
    // Go through all images on the output and for each of them create accumulators and compute loss
    for (int b = 0; b < this->_labels->shape(0); ++b) this->_b_queue.push(b);
    // Wait for the threadpool to finish processing the images
    this->_num_processed.waitToCount(batch_size);

    // Loss per pixel
    Dtype loss = (this->_loss_prob/batch_size + this->_loss_coord/batch_size) / Dtype(2.0f);

    std::cout << "loss_prob: " << (this->_loss_prob / batch_size) << ", loss_coord: " << (this->_loss_coord / batch_size) << std::endl;

    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top,
                                                          const vector<bool> &propagate_down,
                                                          const vector<Blob<Dtype>*> &bottom)
{
    // Fill in the diff (gradient) for each output blob
    if (propagate_down[1])
    {
        // Scale the gradient - they scale it everywhere with batch size (bottom[i]->shape(0)), I don't know
        // why but lets do it as well
        const Dtype alpha = top[0]->cpu_diff()[0] / bottom[1]->shape(0);
        caffe_cpu_axpby(bottom[1]->count(), alpha, this->_diff->cpu_data(), Dtype(0.0f),
                        bottom[1]->mutable_cpu_diff());
    }
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::InternalThreadpoolEntry (int t)
{
    // This method runs on each thread of the internal threadpool
    // We build accumulator, remove negative cooordinate diff and apply diff weights for each image in
    // the batch

    try {
        while (!this->must_stopt(t))
        {
            int b = this->_b_queue.pop();

            // Process this image
            this->_buildAccumulator(b);

            const int batch_size    = this->_bottom->shape(0);
            const int count         = this->_bottom->count() / batch_size;
            const int count_channel = this->_bottom->shape(2) * this->_bottom->shape(3);

            // Compute the difference of the target accumulator and the estimated one
            caffe_sub(count, this->_bottom->cpu_data() + this->_bottom->offset(b, 0),
                             this->_accumulator->cpu_data() + this->_accumulator->offset(b, 0),
                             this->_diff->mutable_cpu_data() + this->_diff->offset(b, 0));

            // We do not want to include errors on coordinates from samples (pixels), which are not supposed
            // to predict them - we need to remove the computed difference from the loss computation
            int num_removed_coords = this->_removeNegativeCoordinateDiff(b);


            // Loss from the probability accumulator (channel 0)
            const Dtype *data_prob = this->_diff->cpu_data() + this->_diff->offset(b, 0);
            Dtype loss_prob = caffe_cpu_dot(count_channel, data_prob, data_prob);
            // Loss from the coordinates (channels 1-7)
            const Dtype *data_coord = this->_diff->cpu_data() + this->_diff->offset(b, 1);
            Dtype loss_coord = caffe_cpu_dot(7*count_channel, data_coord, data_coord);

            // Loss per pixel
            this->_loss_prob  = this->_loss_prob  + loss_prob  / count_channel;
            if (num_removed_coords != 7*count_channel)
            {
                this->_loss_coord = this->_loss_coord + loss_coord / (7*count_channel - num_removed_coords);
            }


            // In the training pass we also have to adjust the diffs (for testing we do not need this)
            if (this->phase_ == TRAIN)
            {
                // Apply weight on the diffs to compensate for the smaller amount of positive pixels in
                // the target accumulator
                this->_applyDiffWeights(b);
            }


            // Raise the counter on processed images
            this->_num_processed.increase();

        }
    } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
    }
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

template <typename Dtype>
void BB3TXTLossLayer<Dtype>::_computeSizeBounds ()
{
    // The bounding boxes that should be detected by a given accumulator are determined by their size. Each
    // accumulator detects bounding boxes with size in some interval <min bound, max bound>. Here we compute
    // these bounds. These bounds are also not sharp, but they overlap so some objects will be detected by
    // multiple accumulators

    const int radius = this->layer_param_.accumulator_loss_param().radius();
    const float cr   = this->layer_param_.accumulator_loss_param().circle_ratio();
    const float bo   = this->layer_param_.accumulator_loss_param().bounds_overlap();

    this->_bb_bounds = std::make_pair(0.0f, 99999.9f);

    // The "ideal" size of a bounding box that should be detected by this accumulator
    this->_ideal_size = float(2*radius+1) / cr * this->_scale;

    // Bound above
    const float ext_above  = ((1-bo)*this->_ideal_size)/2 + bo*this->_ideal_size;
    this->_bb_bounds.second = this->_ideal_size + ext_above;

    // Bound below
    const float difference = this->_ideal_size / 2.0f;
    const float ext_below  = ((1-bo)*difference)/2 + bo*difference;
    this->_bb_bounds.first = this->_ideal_size - ext_below;


    LOG(INFO) << "Scale: x" << this->_scale << ", bbs: " << this->_bb_bounds.first << " to "
              << this->_bb_bounds.second << " px, ideal: " << this->_ideal_size << " px";
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::_buildAccumulator (int b)
{
    // This is because of the cv::Mat type - I don't want to program other types now except CV_32FC1
    CHECK((std::is_same<Dtype,float>::value)) << "BB3TXTLossLayer supports only float!";

    // Radius of the circle drawn in the center of the bounding box
    const int radius = this->layer_param_.accumulator_loss_param().radius();

    const int height = this->_accumulator->shape(2);
    const int width  = this->_accumulator->shape(3);

    Dtype *accumulator_data = this->_accumulator->mutable_cpu_data();

    const double scaling_ratio = 1.0 / this->_scale;


    // The channels of the accumulator go like this: probability, fblx, fbly, fbrx, fbry, rblx, rbly, ftly
    for (int c = 0; c < 8; ++c)
    {
        // Create a cv::Mat wrapper for the accumulator - this way we can now use OpenCV drawing functions
        // to create circles in the accumulator
        cv::Mat acc(height, width, CV_32FC1, accumulator_data + this->_accumulator->offset(b, c));
        acc.setTo(cv::Scalar(0));

        // Draw circles in the center of the bounding boxes
        for (int i = 0; i < this->_labels->shape(1); ++i)
        {
            // Data are stored: [label, xmin, ymin, xmax, ymax, fblx, fbly, fbrx, fbry, rblx, rbly, ftly]
            const Dtype *data = this->_labels->cpu_data() + this->_labels->offset(b, i);

            // If the label is -1, there are no more bounding boxes
            if (data[0] == Dtype(-1.0f)) break;

            // Check if the size of the bounding box fits within the bounds of this accumulator - the
            // largest dimension of the bounding box has to by within the bounds
            Dtype largest_dim = std::max(data[3]-data[1], data[4]-data[2]);
            if (largest_dim >= this->_bb_bounds.first && largest_dim <= this->_bb_bounds.second)
            {
                const int x = (data[1]+data[3]) / 2;
                const int y = (data[2]+data[4]) / 2;

                // Either 1 for probability or coordinates relative to the object centroid
                if (c == 0)
                {
                    cv::circle(acc, cv::Point(scaling_ratio*x,
                                              scaling_ratio*y), radius, cv::Scalar(Dtype(1.0f)), -1);
                }
                else
                {
                    // Render fblx, fbly, fbrx, fbry, rblx, rbly, ftly into the accumulators, that is why +4
                    this->_renderCoordinateCircle(acc, x, y, data[c+4], c);
                }
            }
        }
    }
}


template <typename Dtype>
int BB3TXTLossLayer<Dtype>::_removeNegativeCoordinateDiff (int b)
{
    int num_removed = 0;

    // The channels of the accumulator go like this: probability, fblx, fbly, fbrx, fbry, rblx, rbly, ftly -
    // we want to nullify the diffs of the coordinates, thus indices 1 to 7
    for (int c = 1; c < 8; ++c)
    {
        const Dtype *data_acc_prob = this->_accumulator->cpu_data() + this->_accumulator->offset(b, 0);
        Dtype *data_diff_coord     = this->_diff->mutable_cpu_data() + this->_diff->offset(b, c);

        for (int i = 0; i < this->_accumulator->shape(2)*this->_accumulator->shape(3); ++i)
        {
            // This is a negative sample - nullify coordinate diff
            if (*data_acc_prob == Dtype(0.0f))
            {
                *data_diff_coord = Dtype(0.0f);
                num_removed++;
            }

            data_acc_prob++;
            data_diff_coord++;
        }
    }

    return num_removed;
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::_applyDiffWeights (int b)
{
    // Applies weights on the diffs in order to even the impact of the positive and negative samples on
    // the gradient

    const int count_channel = this->_accumulator->shape(2) * this->_accumulator->shape(3);
    const float nr          = this->layer_param().accumulator_loss_param().negative_ratio();
//    const Dtype HN_THRESH   = 0.25f;


    const Dtype* data_acc_prob  = this->_accumulator->cpu_data() + this->_accumulator->offset(b, 0);
//    const Dtype *data_diff_prob = this->_diff->cpu_data() + this->_diff->offset(b, 0);
    std::vector<Dtype*> data_diff_m;
    for (int c = 0; c < this->_diff->shape(1); ++c)
    {
        data_diff_m.push_back(this->_diff->mutable_cpu_data() + this->_diff->offset(b, c));
    }

    // Number of positive pixels (samples) in this accumulator
    const float pn = caffe_cpu_asum(count_channel, data_acc_prob);
    const float nn = count_channel - pn;

    // Compute the number of hard negative pixels (samples)
//    float hnn = 0.0f;
//    for (int i = 0; i < count_channel; ++i)
//    {
//        // We do not have to check if this is a positive or negative sample because we will never have
//        // a huge positive error on positive samples - the network learns to predict values in [0, 1],
//        // therefore it can only overshoot a value for negative samples, which should be zero. The diff
//        // is therefore positive
//        if (*data_diff_prob > HN_THRESH) hnn++;  // This is a hard negative
//        data_diff_prob++;
//    }

    // Now determine the weight of positive pixels from the required ratio
    const float pos_diff_weight = nn / pn / nr;
    // The weight of negative pixels is adjusted so that hard negatives have the same weigth sum as
    // the other samples' diffs
//    const float neg_diff_weight  = (hnn == 0) ? 1.0f : (nn / (nn-hnn) / 2.0f);
//    const float hneg_diff_weight = (hnn == 0) ? 1.0f : (nn / hnn / 2.0f);

    for (int i = 0; i < count_channel; ++i)
    {
        if (*data_acc_prob > Dtype(0.0f))
        {
            // Positive sample
            for (int c = 0; c < data_diff_m.size(); ++c)
            {
                data_diff_m[c][i] *= pos_diff_weight;
            }
        }
//        else if (*data_diff_prob_m > HN_THRESH)
//        {
//            // Hard negative sample
//            *data_diff_prob_m *= hneg_diff_weight;
//        }
//        else
//        {
//            // Other negative samples
//            *data_diff_prob_m *= neg_diff_weight;
//        }

        data_acc_prob++;
    }
}


template <typename Dtype>
void BB3TXTLossLayer<Dtype>::_renderCoordinateCircle (cv::Mat &acc, int x, int y, Dtype value, int channel)
{
    // The circles in the accumulators with the coordinates have to be rendered separately because the value
    // of each pixel differs based on the pixel position inside of the bounding box

    const Dtype DUMMY = 9999.0f;
    const int radius  = this->layer_param_.accumulator_loss_param().radius();
    const int x_acc   = x / this->_scale;
    const int y_acc   = y / this->_scale;

    // Because I don't want to take care of the plotting of the circle myself, I use the OpenCV function and
    // then replace the values - first create a circle with a dummy value
    cv::circle(acc, cv::Point(x_acc, y_acc), radius, cv::Scalar(DUMMY), -1);

    // Now go through the pixels in the circle's bounding box and if there is the DUMMY value compute
    // the real value
    for (int i = -radius; i <= radius; ++i)
    {
        for (int j = -radius; j <= radius; ++j)
        {
            int xp = x_acc + j;
            int yp = y_acc + i;

            if (xp >= 0 && xp < acc.cols && yp >= 0 && yp < acc.rows)
            {
                // Pixel is inside of the accumulator - check if it contains DUMMY
                if (acc.at<Dtype>(yp, xp) == DUMMY)
                {
                    // Change its value to the actual value - coordinate relative to the pixel position
                    // The coordinates are converted to approximately [0,1], i.e. the ideal bounding box has
                    // coordinates [0,0], [1,1], but the actual bounding boxes will differ in both dimensions
                    // label, fblx, fbly, fbrx, fbry, rblx, rbly, ftly
                    if (channel == 1 || channel == 3 || channel == 5)
                    {
                        // fblx, fbrx, rblx
                        acc.at<Dtype>(yp, xp) = Dtype(0.5f + (value-x - j*this->_scale) / this->_ideal_size);
                    }
                    else if (channel == 2 || channel == 4 || channel == 6 || channel == 7)
                    {
                        // fblx, fbly, fbrx, fbry, rblx, rbly, ftly
                        acc.at<Dtype>(yp, xp) = Dtype(0.5f + (value-y - i*this->_scale) / this->_ideal_size);
                    }
                }
            }
        }
    }
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(BB3TXTLossLayer);
#endif

INSTANTIATE_CLASS(BB3TXTLossLayer);
REGISTER_LAYER_CLASS(BB3TXTLoss);


}  // namespace caffe
