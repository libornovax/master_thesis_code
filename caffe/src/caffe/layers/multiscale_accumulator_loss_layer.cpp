#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/multiscale_accumulator_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {


template <typename Dtype>
MultiscaleAccumulatorLossLayer<Dtype>::MultiscaleAccumulatorLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param),
      _rng(new Caffe::RNG(caffe_rng_rand()))
{
    CHECK(param.has_accumulator_loss_param()) << "AccumulatorLossParameter is mandatory!";
    CHECK(param.accumulator_loss_param().has_radius()) << "Radius is mandatory!";
    CHECK(param.accumulator_loss_param().has_circle_ratio()) << "Circle ratio is mandatory!";
    CHECK(param.accumulator_loss_param().has_downsampling()) << "Downsampling is mandatory!";
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Reshape (const vector<Blob<Dtype>*> &bottom,
                                           const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first and the accumulators follow

    LossLayer<Dtype>::Reshape(bottom, top);

    const int num_accumulators = bottom.size() - 1;
    int scale = this->layer_param_.accumulator_loss_param().downsampling();

    if (num_accumulators != this->_accumulators.size() || num_accumulators != this->_diffs.size())
    {
        // This should only happen once during initialization -> create blobs for accumulators
        this->_accumulators.clear();
        this->_diffs.clear();
        this->_scales.clear();
        this->_bb_bounds.clear();

        for (int i = 0; i < num_accumulators; ++i)
        {
            this->_accumulators.push_back(std::make_shared<Blob<Dtype>>(bottom[i+1]->shape()));
            this->_diffs.push_back(std::make_shared<Blob<Dtype>>(bottom[i+1]->shape()));

            // Compute the scales of the accumulators
            this->_scales.push_back(scale);

            // Compute the bounds for bounding boxes, which should be included in this accumulator
            this->_bb_bounds.push_back(this->_computeSizeBounds(i, num_accumulators, scale));

            scale *= 2;
        }
    }
    else
    {
        // Only reshape the blobs - batch size could have changed
        for (int i = 0; i < num_accumulators; ++i)
        {
            this->_accumulators[i]->ReshapeLike(*bottom[i+1]);
            this->_diffs[i]->ReshapeLike(*bottom[i+1]);
        }
    }
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom,
                                               const vector<Blob<Dtype>*> &top)
{
    // Create the accumulators from the labels
    this->_buildAccumulators(bottom[0]);

    const float nr = this->layer_param_.accumulator_loss_param().negative_ratio();
    Dtype loss = Dtype(0.0f);

    // Now it is a simple squared Euclidean distance of the net outputs and the accumulators
    int num_accumulators = bottom.size() - 1;
    for (int i = 0; i < num_accumulators; ++i)
    {
        // Store the current output - currently processed bottom blob (i=0 are labels)
        const Blob<Dtype> *output = bottom[i+1];
        int count = output->count();
        caffe_sub(count, output->cpu_data(), this->_accumulators[i]->cpu_data(),
                  this->_diffs[i]->mutable_cpu_data());

#ifdef USE_DIFF_WEIGHT
        // Apply weight on the negative samples (pixels) to compensate for the smaller amount of positive
        // pixels in the target accumulator
        float neg_diff_weight = 1.0f / (float(count) / nr / output->shape(0));
        Dtype num_active = this->_applyDiffWeights(i, neg_diff_weight);
#else
        // Apply mask on this diff - we only want to include some number of negative pixels because otherwise
        // the learning would be skewed towards learning mostly negative examples
        Dtype num_active = this->_applyMask(i);
#endif

        Dtype dot = caffe_cpu_dot(count, this->_diffs[i]->cpu_data(), this->_diffs[i]->cpu_data());
        loss += dot / output->shape(0) / num_active / Dtype(2); // Per image, per pixel loss
    }

    // Write the loss to the output blob
    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top,
                                                const vector<bool> &propagate_down,
                                                const vector<Blob<Dtype>*> &bottom)
{
    int num_accumulators = bottom.size() - 1;
    for (int i = 0; i < num_accumulators; ++i)
    {
        // Fill in the diff for each output accumulator
        if (propagate_down[i+1])
        {
            const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i+1]->shape(0);
            caffe_cpu_axpby(bottom[i+1]->count(), alpha, this->_diffs[i]->cpu_data(), Dtype(0),
                            bottom[i+1]->mutable_cpu_diff());
        }
    }
}



// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

template <typename Dtype>
std::pair<float, float>
MultiscaleAccumulatorLossLayer<Dtype>::_computeSizeBounds (const int i, const int num_accumulators,
                                                           const int scale) const
{
    // The bounding boxes that should be detected by a given accumulator are determined by their size. Each
    // accumulator detects bounding boxes with size in some interval <min bound, max bound>. Here we compute
    // these bounds. These bounds are also not sharp, but they overlap so some objects will be detected by
    // multiple accumulators

    const int radius = this->layer_param_.accumulator_loss_param().radius();
    const float cr   = this->layer_param_.accumulator_loss_param().circle_ratio();
    const float bo   = this->layer_param_.accumulator_loss_param().bounds_overlap();

    auto bounds = std::make_pair(0.0f, 99999.9f);

    // The "ideal" size of a bounding box that should be detected by this accumulator
    const float ideal_size = float(2*radius+1) / cr * scale;

    if (i < num_accumulators-1 && num_accumulators != 1)
    {
        // Everything but the last accumulator will have a bound above
        const float ext_above  = ((1-bo)*ideal_size)/2 + bo*ideal_size;
        bounds.second = ideal_size + ext_above;
    }
    if (i > 0 && num_accumulators != 1)
    {
        // Everything but the first accumulator will have a bound below
        const float difference = ideal_size / 2.0f;
        const float ext_below  = ((1-bo)*difference)/2 + bo*difference;
        bounds.first = ideal_size - ext_below;
    }

    LOG(INFO) << "[" << i << "] Scale: x" << scale << ", bbs: " << bounds.first << " to "
              << bounds.second << " px, ideal: " << ideal_size << " px";

    return bounds;
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::_buildAccumulators (const Blob<Dtype> *labels)
{
    // This is because of the cv::Mat type - I don't want to program other types now except CV_32FC1
    CHECK((std::is_same<Dtype,float>::value)) << "MultiscaleAccumulatorLossLayer supports only float!";

    // Radius of the circle drawn in the center of the bounding box
    const int radius = this->layer_param_.accumulator_loss_param().radius();

    // Fill accumulators of all scales
    for (int a = 0; a < this->_accumulators.size(); ++a)
    {
        std::shared_ptr<Blob<Dtype>> accumulator = this->_accumulators[a];

        const int height = accumulator->shape(2);
        const int width  = accumulator->shape(3);

        Dtype *accumulator_data = accumulator->mutable_cpu_data();

        const double scaling_ratio = 1.0 / this->_scales[a];

        // Go through all images on the output and for each of them create accumulators
        for (int b = 0; b < labels->shape(0); ++b)
        {
            // Create a cv::Mat wrapper for the accumulator - this way we can now use OpenCV drawing functions
            // to create circles in the accumulator
            cv::Mat acc(height, width, CV_32FC1, accumulator_data + accumulator->offset(b));
            acc.setTo(cv::Scalar(0));

            // Draw circles in the center of the bounding boxes
            for (int i = 0; i < labels->shape(1); ++i)
            {
                // Data are stored like this [label, xmin, ymin, xmax, ymax]
                const Dtype *data = labels->cpu_data() + labels->offset(b, i);

                // If everything is 0, there are no more bounding boxes
                if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 0 && data[4] == 0) break;

                // Check if the size of the bounding box fits within the bounds of this accumulator - the
                // largest dimension of the bounding box has to by within the bounds
                Dtype largest_dim = std::max(data[3]-data[1], data[4]-data[2]);
                if (largest_dim > this->_bb_bounds[a].first && largest_dim < this->_bb_bounds[a].second)
                {
                    cv::circle(acc, cv::Point(scaling_ratio*(data[1]+data[3])/2,
                                              scaling_ratio*(data[2]+data[4])/2), radius, cv::Scalar(1), -1);
                }
            }
        }
    }
}


#ifndef USE_DIFF_WEIGHT
template <typename Dtype>
Dtype MultiscaleAccumulatorLossLayer<Dtype>::_applyMask (int i)
{
    // Apply mask - we only want to include the given ratio of negative pixels with respect to the positive
    // ones because otherwise the learning would be skewed towards learning mostly negative examples. Thus
    // we create a random mask on the negative examples to ensure the negative:positive ratio of samples
    // given by the layer parameters

    const std::shared_ptr<Blob<Dtype>> accumulator = this->_accumulators[i];

    CHECK_EQ(accumulator->count(), this->_diffs[i]->count()) << "Accumulator and diff size is not the same!";

    // The number of positive samples is the number of 1s in the accumulator
    const int num_positive = caffe_cpu_asum(accumulator->count(), accumulator->cpu_data());
    const int num_negative = num_positive * this->layer_param_.accumulator_loss_param().negative_ratio();

    caffe::rng_t* rng = static_cast<caffe::rng_t*>(this->_rng->generator());
    boost::random::uniform_int_distribution<> dist(0, accumulator->count());

    Dtype* data_diff              = this->_diffs[i]->mutable_cpu_data();
    const Dtype* data_accumulator = accumulator->cpu_data();

    for (int j = 0; j < accumulator->count(); ++j)
    {
        // If this is a negative sample -> randomly choose if we mask this diff out or not
        if (*data_accumulator == Dtype(0.0f) && dist(*rng) > num_negative)
        {
            // Mask it out
            *data_diff = Dtype(0.0f);
        }

        data_diff++;
        data_accumulator++;
    }

    // The number of active samples (pixels) is the sum of positive and negative ones
    return Dtype(num_positive + num_negative);
}
#else
template <typename Dtype>
Dtype MultiscaleAccumulatorLossLayer<Dtype>::_applyDiffWeights (int i, float neg_diff_weight)
{
    // Applies weights on the diffs of the negative samples in order to decrease their significance in
    // the loss function

    const std::shared_ptr<Blob<Dtype>> accumulator = this->_accumulators[i];

    CHECK_EQ(accumulator->count(), this->_diffs[i]->count()) << "Accumulator and diff size is not the same!";

    Dtype* data_diff              = this->_diffs[i]->mutable_cpu_data();
    const Dtype* data_accumulator = accumulator->cpu_data();

    for (int j = 0; j < accumulator->count(); ++j)
    {
        // If this is a negative sample -> multiply its diff
        if (*data_accumulator == Dtype(0.0f))
        {
            *data_diff *= neg_diff_weight;
        }

        data_diff++;
        data_accumulator++;
    }

    // The number of active samples (pixels) is the sum of positive and negative ones, which in this case is
    // accumulator->count(), i.e. number of pixels in the accumulator
    return Dtype(accumulator->count());
}
#endif


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(MultiscaleAccumulatorLossLayer);
#endif

INSTANTIATE_CLASS(MultiscaleAccumulatorLossLayer);
REGISTER_LAYER_CLASS(MultiscaleAccumulatorLoss);


}  // namespace caffe
