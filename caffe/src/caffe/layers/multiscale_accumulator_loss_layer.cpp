#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/multiscale_accumulator_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {


template <typename Dtype>
MultiscaleAccumulatorLossLayer<Dtype>::MultiscaleAccumulatorLossLayer(const LayerParameter &param)
    : LossLayer<Dtype>(param)
{
    CHECK(param.has_accumulator_loss_param()) << "AccumulatorLossParameter is mandatory!";
    CHECK(param.accumulator_loss_param().has_radius()) << "Radius is mandatory!";
    CHECK(param.accumulator_loss_param().has_circle_ratio()) << "Circle ratio is mandatory!";
    CHECK(param.accumulator_loss_param().has_downsampling()) << "Downsampling is mandatory!";
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>*> &bottom,
                                                        const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first and the accumulators follow

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    const int num_accumulators = bottom.size() - 1;
    int scale = this->layer_param_.accumulator_loss_param().downsampling();

    this->_accumulators.clear();
    this->_diffs.clear();
    this->_scales.clear();
    this->_bb_bounds.clear();

    this->_sigmoid_layers.clear();
    this->_sigmoid_outputs.clear();
    this->_sigmoid_bottom_vecs.clear();
    this->_sigmoid_top_vecs.clear();

    for (int i = 0; i < num_accumulators; ++i)
    {
        this->_accumulators.push_back(std::make_shared<Blob<Dtype>>());
        this->_diffs.push_back(std::make_shared<Blob<Dtype>>());
        // Compute the scales of the accumulators
        this->_scales.push_back(scale);
        // Compute the bounds for bounding boxes, which should be included in this accumulator
        this->_bb_bounds.push_back(this->_computeSizeBounds(i, num_accumulators, scale));
        scale *= 2;

        // Initialize the sigmoid layers - see sigmoid_cross_entropy_loss_layer for details
        this->_sigmoid_layers.emplace_back(new SigmoidLayer<Dtype>(this->layer_param()));
        this->_sigmoid_outputs.emplace_back(new Blob<Dtype>());
        this->_sigmoid_bottom_vecs.emplace_back(); // This creates an empty vector
        this->_sigmoid_bottom_vecs[i].push_back(bottom[i+1]);
        this->_sigmoid_top_vecs.emplace_back(); // This creates an empty vector
        this->_sigmoid_top_vecs[i].push_back(this->_sigmoid_outputs[i].get());
        this->_sigmoid_layers[i]->SetUp(this->_sigmoid_bottom_vecs[i], this->_sigmoid_top_vecs[i]);
    }
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Reshape (const vector<Blob<Dtype>*> &bottom,
                                                     const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first and the accumulators follow

    LossLayer<Dtype>::Reshape(bottom, top);

    const int num_accumulators = bottom.size() - 1;
    for (int i = 0; i < num_accumulators; ++i)
    {
        this->_accumulators[i]->ReshapeLike(*bottom[i+1]);
        this->_diffs[i]->ReshapeLike(*bottom[i+1]);

        this->_sigmoid_layers[i]->Reshape(this->_sigmoid_bottom_vecs[i], this->_sigmoid_top_vecs[i]);

        CHECK_EQ(this->_sigmoid_top_vecs[i][0]->count(), bottom[i+1]->count())
                << "Sigmoid layer must have the same size as accumulator!";
    }
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom,
                                                         const vector<Blob<Dtype>*> &top)
{
    // Create the accumulators from the labels
    this->_buildAccumulators(bottom[0]);

    Dtype loss_total  = Dtype(0.0f);
    const int num_acc = bottom.size() - 1;

    // Now it is a simple squared Euclidean distance of the net outputs and the accumulators
    for (int i = 0; i < num_acc; ++i)
    {
        // Compute the sigmoid outputs - they are actually not used here, but in the backward pass to compute
        // the diffs
        this->_sigmoid_bottom_vecs[i][0] = bottom[i+1];
        this->_sigmoid_layers[i]->Forward(this->_sigmoid_bottom_vecs[i], this->_sigmoid_top_vecs[i]);

        const int count = bottom[i+1]->count();
        Dtype loss      = Dtype(0.0f);

        // Compute the loss (negative log likelihood)
        const Dtype* output = bottom[i+1]->cpu_data();
        const Dtype* target = this->_accumulators[i]->cpu_data();
        for (int j = 0; j < count; ++j)
        {
            loss -= output[j] * (target[j] - (output[j] >= 0)) -
                    log(1 + exp(output[j] - 2 * output[j] * (output[j] >= 0)));
        }

        loss_total += loss / count;
    }

    // Write the loss to the output blob
    top[0]->mutable_cpu_data()[0] = loss_total / num_acc;
}


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top,
                                                          const vector<bool> &propagate_down,
                                                          const vector<Blob<Dtype>*> &bottom)
{
    const int num_acc = bottom.size() - 1;
    for (int i = 0; i < num_acc; ++i)
    {
        // Fill in the diff for each output accumulator
        if (propagate_down[i+1])
        {
            const int count = bottom[i+1]->count();

            caffe_sub(count, this->_sigmoid_outputs[i]->cpu_data(), this->_accumulators[i]->cpu_data(),
                      bottom[i+1]->mutable_cpu_diff());

            // Apply weight on the diffs to compensate for the smaller amount of positive pixels in
            // the target accumulator
            this->_applyDiffWeights(i, bottom[i+1]);

            // Scale the gradient
            // CAREFUL HERE! We cannot scale it too much otherwise the net will basically have such small
            // gradients that it won't learn anything!
            const Dtype loss_weight = top[0]->cpu_diff()[0] / (count / bottom[i+1]->shape(0));
            caffe_scal(count, loss_weight, bottom[i+1]->mutable_cpu_diff());
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


template <typename Dtype>
void MultiscaleAccumulatorLossLayer<Dtype>::_applyDiffWeights (int i, Blob<Dtype> *bottom)
{
    // Applies weights on the diffs in order to even the impact of the positive and negative samples on
    // the gradient

    const std::shared_ptr<Blob<Dtype>> accumulator = this->_accumulators[i];
    CHECK_EQ(accumulator->count(), bottom->count()) << "Accumulator and diff size is not the same!";

    Dtype* data_diff              = bottom->mutable_cpu_data();
    const Dtype* data_accumulator = accumulator->cpu_data();

    const float pn = caffe_cpu_asum(accumulator->count(), accumulator->cpu_data()) / bottom->shape(0);
    const float nr = this->layer_param().accumulator_loss_param().negative_ratio();
    float pos_diff_weight = accumulator->count() / nr / bottom->shape(0) / pn;

    for (int j = 0; j < accumulator->count(); ++j)
    {
        if (*data_accumulator > Dtype(0.0f))
        {
            *data_diff *= pos_diff_weight;
        }

        data_diff++;
        data_accumulator++;
    }
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(MultiscaleAccumulatorLossLayer);
#endif

INSTANTIATE_CLASS(MultiscaleAccumulatorLossLayer);
REGISTER_LAYER_CLASS(MultiscaleAccumulatorLoss);


}  // namespace caffe
