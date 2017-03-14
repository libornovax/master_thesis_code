//
// Libor Novak
// 03/13/2017
//

#ifndef CAFFE_BBTXT_LOSS_LAYER_HPP_
#define CAFFE_BBTXT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"


namespace caffe {


/**
 * @brief The BBTXTLossLayer class
 *
 * The loss is composed of two different losses - Accumulator loss and regression loss from bounding box
 * coordinates. This loss supports only one accumulator! I we want multiple accumulators in the network, each
 * accumulator has to have its own bbtxt loss layer
 */
template <typename Dtype>
class BBTXTLossLayer : public LossLayer<Dtype>
{
public:

    explicit BBTXTLossLayer (const LayerParameter &param);

    virtual void LayerSetUp (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

    virtual void Reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;


    // -----------------------------------------  INLINE METHODS  ---------------------------------------- //

    virtual inline const char* type () const override
    {
        return "BBTXTLoss";
    }


protected:

    virtual void Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;
//    virtual void Forward_gpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

    virtual void Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                               const vector<Blob<Dtype>*> &bottom) override;
//    virtual void Backward_gpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
//                               const vector<Blob<Dtype>*> &bottom) override;

    /**
     * @brief Computes the bounds of bounding box sizes for the accumulator
     */
    virtual void _computeSizeBounds ();

    /**
     * @brief Builds the accumulators from the given labels
     * @param labels Blob with the labels of shape batch x num_bbs x 5
     */
    virtual void _buildAccumulator (const Blob<Dtype> *labels);

    /**
     * @brief Removes diffs of coordinates of negative samples - we do not want to include them in the loss
     * @return Number of removed coordinates
     */
    virtual int _removeNegativeCoordinateDiff ();

    /**
     * @brief Weights the diffs in order to even out the impact of positive and negative samples on the gradient
     */
    virtual void _applyDiffWeights ();


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // Hough map - accumulator of object centers (shape of the blob is the same as bottom[0])
    std::shared_ptr<Blob<Dtype>> _accumulator;
    // Scale (downsampling) of the accumulator
    int _scale;
    // Each accumulator only includes some bounding boxes - given the size of the boundig box. The _bb_bounds
    // contain bounds on the bounding box sizes, which are going to be included in this accumulator
    std::pair<float,float> _bb_bounds;
    float _ideal_size;

    std::shared_ptr<Blob<Dtype>> _diff;
};


}  // namespace caffe


#endif  // CAFFE_BBTXT_LOSS_LAYER_HPP_
