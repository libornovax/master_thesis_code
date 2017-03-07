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
#include "caffe/layers/sigmoid_layer.hpp"


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

    virtual void LayerSetUp (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

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
     * @brief Weights the diffs in order to even out the impact of positive and negative samples on the gradient
     * @param i Index of the accumulator and diff, which will be weighted
     * @param bottom The bottom blob, whose diffs will be altered
     */
    void _applyDiffWeights (int i, Blob<Dtype> *bottom);


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

    /// The internal SigmoidLayer used to map predictions to probabilities.
    std::vector<std::shared_ptr<SigmoidLayer<Dtype>>> _sigmoid_layers;
    /// sigmoid_output stores the output of the SigmoidLayer.
    std::vector<std::shared_ptr<Blob<Dtype>>> _sigmoid_outputs;
    /// bottom vector holder to call the underlying SigmoidLayer::Forward
    std::vector<std::vector<Blob<Dtype>*>> _sigmoid_bottom_vecs;
    /// top vector holder to call the underlying SigmoidLayer::Forward
    std::vector<std::vector<Blob<Dtype>*>> _sigmoid_top_vecs;

};


}  // namespace caffe


#endif  // CAFFE_MULTISCALE_ACCUMULATOR_LOSS_LAYER_HPP_
