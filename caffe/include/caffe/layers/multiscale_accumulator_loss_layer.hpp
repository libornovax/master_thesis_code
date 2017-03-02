//
// Libor Novak
// 02/25/2017
//

#ifndef CAFFE_MULTISCALE_ACCUMULATOR_LOSS_LAYER_HPP_
#define CAFFE_MULTISCALE_ACCUMULATOR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

// Instead of random selection of samples the positive samples are weighted to even their amount in
// the ground truth accumulator
#define USE_DIFF_WEIGHT


namespace caffe {


/**
 * @brief The MultiscaleAccumulatorLossLayer class
 *
 * This is a similar loss function to AccumulatorLossLayer, however this one assumes that there are bounding
 * boxes of different sizes in the dataset. There are multiple accumulators for different object scales
 * and given the bounding box size, a circle is created in an accumulator that best fits the object size.
 */
template <typename Dtype>
class MultiscaleAccumulatorLossLayer : public LossLayer<Dtype>
{
public:

    explicit MultiscaleAccumulatorLossLayer (const LayerParameter &param);

    virtual void Reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;


    // -----------------------------------------  INLINE METHODS  ---------------------------------------- //

    virtual inline const char* type () const override
    {
        return "MultiscaleAccumulatorLoss";
    }

    virtual inline int ExactNumBottomBlobs() const override
    {
        // We can compute the loss on any amount of accumulators
        return -1;
    }


protected:

    virtual void Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;
//    virtual void Forward_gpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

    virtual void Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                               const vector<Blob<Dtype>*> &bottom) override;
//    virtual void Backward_gpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
//                               const vector<Blob<Dtype>*> &bottom) override;

    /**
     * @brief Computes the bounds of bounding box sizes for the given accumulator
     * @param i Index of the accumulator
     * @param num_accumulators Total number of accumulators in the layer
     * @param scale Scaling factor of the given accumulator
     * @return Output variable that will be filled with the bounds
     */
    std::pair<float, float> _computeSizeBounds (const int i, const int num_accumulators,
                                                const int scale) const;

    /**
     * @brief Builds the accumulators from the given labels
     * @param labels Blob with the labels of shape batch x num_bbs x 5
     */
    void _buildAccumulators (const Blob<Dtype> *labels);

    /**
     * @brief Computes the weight of positive samples in the current batch
     * The weight which will be used to multiply the diffs of positive samples to assure the required ratio
     * of negative:postive samples in the batch
     * @return weight
     */
    float _computePosDiffWeight () const;

#ifndef USE_DIFF_WEIGHT
    /**
     * @brief Applies a random mask to a diff to ensure the required ratio of positive and negative samples
     * @param i Index of the accumulator and diff to which apply the mask
     * @return Number of active samples
     */
    Dtype _applyMask (int i);
#else
    /**
     * @brief Weights the positive samples to increase their impact in the cost function and learning
     * @param i Index of the accumulator and diff, which will be weighted
     * @param pos_diff_weight Weight to be applied to positive samples
     * @return Number of active samples (size of the accumulator)
     */
    Dtype _applyDiffWeights (int i, float pos_diff_weight);
#endif


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // Hough map - accumulator of object centers (shape of the blob is the same as bottom[0])
    std::vector<std::shared_ptr<Blob<Dtype>>> _accumulators;
    // Diff of the accumulator and the net output - saved for backpropagation
    std::vector<std::shared_ptr<Blob<Dtype>>> _diffs;
    // Scales (downsamplings) of the accumulators
    std::vector<int> _scales;
    // Each accumulator only includes some bounding boxes - given the size of the boundig box. The _bb_bounds
    // contain bounds on the bounding box sizes, which are going to be included in the respective accumulator
    std::vector<std::pair<float,float>> _bb_bounds;
    // Random number generator
    shared_ptr<Caffe::RNG> _rng;

};


}  // namespace caffe


#endif  // CAFFE_MULTISCALE_ACCUMULATOR_LOSS_LAYER_HPP_
