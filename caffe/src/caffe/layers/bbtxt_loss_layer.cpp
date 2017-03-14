#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/bbtxt_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {


template <typename Dtype>
BBTXTLossLayer<Dtype>::BBTXTLossLayer(const LayerParameter &param)
    : LossLayer<Dtype>(param)
{
    CHECK(param.has_accumulator_loss_param()) << "AccumulatorLossParameter is mandatory!";
    CHECK(param.accumulator_loss_param().has_radius()) << "Radius is mandatory!";
    CHECK(param.accumulator_loss_param().has_circle_ratio()) << "Circle ratio is mandatory!";
    CHECK(param.accumulator_loss_param().has_downsampling()) << "Downsampling is mandatory!";
}


template <typename Dtype>
void BBTXTLossLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first bottom[0] and then the accumulator bottom[1]

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    this->_accumulator = std::make_shared<Blob<Dtype>>();
    this->_diff        = std::make_shared<Blob<Dtype>>();

    // Scale of this accumulator is given by downsampling
    this->_scale = this->layer_param_.accumulator_loss_param().downsampling();
    // Compute the bounds for bounding boxes, which should be included in this accumulator
    this->_computeSizeBounds();
}


template <typename Dtype>
void BBTXTLossLayer<Dtype>::Reshape (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    // IMPORTANT: The label blob is first bottom[0] and then the accumulator bottom[1]

    LossLayer<Dtype>::Reshape(bottom, top);

    CHECK_EQ(bottom[1]->shape(1), 5) << "Accumulator must have 5 channels (prob, xmin, ymin, xmax, ymax)!";

    this->_accumulator->ReshapeLike(*bottom[1]);
    this->_diff->ReshapeLike(*bottom[1]);
}


template <typename Dtype>
void BBTXTLossLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    // Create the accumulators from the labels
    this->_buildAccumulator(bottom[0]);

    const int count         = bottom[1]->count();
    const int count_channel = bottom[1]->shape(2) * bottom[1]->shape(3);
    const int batch_size    = bottom[1]->shape(0);

    // Compute the difference of the target accumulator and the estimated one
    caffe_sub(count, bottom[1]->cpu_data(), this->_accumulator->cpu_data(), this->_diff->mutable_cpu_data());

    // We do not want to include errors on coordinates from samples (pixels), which are not supposed to
    // predict them - we need to remove the computed difference from the loss computation
    const int num_removed_coords = this->_removeNegativeCoordinateDiff();

    Dtype loss_prob  = Dtype(0.0f);
    Dtype loss_coord = Dtype(0.0f);
    for (int b = 0; b < batch_size; ++b)
    {
        // Loss from the probability accumulator (channel 0)
        const Dtype *data_prob = this->_diff->cpu_data() + this->_diff->offset(b, 0);
        loss_prob += caffe_cpu_dot(count_channel, data_prob, data_prob);

        // Loss from the coordinates (channels 1-4)
        const Dtype *data_coord = this->_diff->cpu_data() + this->_diff->offset(b, 1);
        loss_coord += caffe_cpu_dot(4*count_channel, data_coord, data_coord);
    }

    // Loss per pixel
    Dtype loss = (loss_prob / (batch_size*count_channel)
                + loss_coord / (4*batch_size*count_channel - num_removed_coords)) / Dtype(2.0f);

    std::cout << "loss_prob: " << (loss_prob / (batch_size*count_channel)) << ", loss_coord: " << (loss_coord / (4*batch_size*count_channel - num_removed_coords)) << std::endl;

    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void BBTXTLossLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top,
                                                          const vector<bool> &propagate_down,
                                                          const vector<Blob<Dtype>*> &bottom)
{
    // Fill in the diff (gradient) for each output blob
    if (propagate_down[1])
    {
        // Apply weight on the diffs to compensate for the smaller amount of positive pixels in
        // the target accumulator
        this->_applyDiffWeights();

        // Scale the gradient - they scale it everywhere with batch size (bottom[i]->shape(0)), I don't know
        // why but lets do it as well
        const Dtype alpha = top[0]->cpu_diff()[0] / bottom[1]->shape(0);
        caffe_cpu_axpby(bottom[1]->count(), alpha, this->_diff->cpu_data(), Dtype(0.0f),
                        bottom[1]->mutable_cpu_diff());
    }
}



// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

template <typename Dtype>
void BBTXTLossLayer<Dtype>::_computeSizeBounds ()
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
void BBTXTLossLayer<Dtype>::_buildAccumulator (const Blob<Dtype> *labels)
{
    // This is because of the cv::Mat type - I don't want to program other types now except CV_32FC1
    CHECK((std::is_same<Dtype,float>::value)) << "BBTXTLossLayer supports only float!";

    // Radius of the circle drawn in the center of the bounding box
    const int radius = this->layer_param_.accumulator_loss_param().radius();

    const int height = this->_accumulator->shape(2);
    const int width  = this->_accumulator->shape(3);

    Dtype *accumulator_data = this->_accumulator->mutable_cpu_data();

    const double scaling_ratio = 1.0 / this->_scale;

    // Go through all images on the output and for each of them create accumulators
    for (int b = 0; b < labels->shape(0); ++b)
    {
        // The channels of the accumulator go like this: probability, xmin, ymin, xmax, ymax
        for (int c = 0; c < 5; ++c)
        {
            // Create a cv::Mat wrapper for the accumulator - this way we can now use OpenCV drawing functions
            // to create circles in the accumulator
            cv::Mat acc(height, width, CV_32FC1, accumulator_data + this->_accumulator->offset(b, c));
            acc.setTo(cv::Scalar(0));

            // Draw circles in the center of the bounding boxes
            for (int i = 0; i < labels->shape(1); ++i)
            {
                // Data are stored like this [label, xmin, ymin, xmax, ymax]
                const Dtype *data = labels->cpu_data() + labels->offset(b, i);

                // If the label is -1, there are no more bounding boxes
                if (data[0] == Dtype(-1.0f)) break;

                // Check if the size of the bounding box fits within the bounds of this accumulator - the
                // largest dimension of the bounding box has to by within the bounds
                Dtype largest_dim = std::max(data[3]-data[1], data[4]-data[2]);
                if (largest_dim >= this->_bb_bounds.first && largest_dim <= this->_bb_bounds.second)
                {
                    const int x = (data[1]+data[3]) / 2;
                    const int y = (data[2]+data[4]) / 2;

                    // Either 1 for probability or coordinate relative to the object centroid. The coordinates
                    // are converted to approximately [0,1], i.e. the ideal bounding box has coordinates
                    // [0,0], [1,1], but the actual bounding boxes will differ in both dimensions
                    cv::Scalar value = cv::Scalar(Dtype(1.0f));
                    if (c == 1 || c == 3)
                    {
                        // xmin, xmax
                        value = cv::Scalar(Dtype(0.5f + (data[c] - x) / this->_ideal_size));
                    }
                    else if (c == 2 || c == 4)
                    {
                        // ymin, ymax
                        value = cv::Scalar(Dtype(0.5f + (data[c] - y) / this->_ideal_size));
                    }

                    cv::circle(acc, cv::Point(scaling_ratio*x, scaling_ratio*y), radius, value, -1);
                }
            }
        }
    }
}


template <typename Dtype>
int BBTXTLossLayer<Dtype>::_removeNegativeCoordinateDiff ()
{
    int num_removed = 0;

    for (int b = 0; b < this->_accumulator->shape(0); ++b)
    {
        // The channels of the accumulator go like this: probability, xmin, ymin, xmax, ymax - we want to
        // nullify the diffs of the coordinates, thus indices 1 to 4
        for (int c = 1; c < 5; ++c)
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
    }

    return num_removed;
}


template <typename Dtype>
void BBTXTLossLayer<Dtype>::_applyDiffWeights ()
{
    // Applies weights on the diffs in order to even the impact of the positive and negative samples on
    // the gradient

    const int count_channel = this->_accumulator->shape(2) * this->_accumulator->shape(3);

    // Compute the number of positive pixels in the probability accumulator
    float pn = 0.0f;
    for (int b = 0; b < this->_accumulator->shape(0); ++b)
    {
        const Dtype* data_acc_prob = this->_accumulator->cpu_data() + this->_accumulator->offset(b, 0);
        pn += caffe_cpu_asum(count_channel, data_acc_prob);
    }
    pn /= this->_accumulator->shape(0);  // Per image
    // Now determine the weight of positive pixels from the required ratio
    const float nr = this->layer_param().accumulator_loss_param().negative_ratio();
    const float pos_diff_weight = (count_channel-pn) / nr / pn;

    std::cout << "pn: " << pn << std::endl;
    std::cout << "pos diff weight: " << pos_diff_weight << std::endl;

    for (int b = 0; b < this->_accumulator->shape(0); ++b)
    {
        const Dtype* data_acc_prob = this->_accumulator->cpu_data() + this->_accumulator->offset(b, 0);
        Dtype *data_diff_prob      = this->_diff->mutable_cpu_data() + this->_diff->offset(b, 0);

        for (int i = 0; i < count_channel; ++i)
        {
            if (*data_acc_prob > Dtype(0.0f))
            {
                *data_diff_prob *= pos_diff_weight;
            }

            data_diff_prob++;
            data_acc_prob++;
        }
    }
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(BBTXTLossLayer);
#endif

INSTANTIATE_CLASS(BBTXTLossLayer);
REGISTER_LAYER_CLASS(BBTXTLoss);


}  // namespace caffe
