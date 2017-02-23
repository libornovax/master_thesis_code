//
// Libor Novak
// 02/22/2017
//

#ifndef CAFFE_ACCUMULATOR_LOSS_LAYER_HPP_
#define CAFFE_ACCUMULATOR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {


/**
 * @brief The AccumulatorLossLayer class
 *
 * Receives bounding box descriptions as labels and creates accumulator out of the bounding boxes. This
 * accumulator is basically a required Hough map (accumulator) - probability map of object centers.
 */
template <typename Dtype>
class AccumulatorLossLayer : public LossLayer<Dtype>
{
public:

    explicit AccumulatorLossLayer (const LayerParameter &param);

    virtual void Reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;


    // -----------------------------------------  INLINE METHODS  ---------------------------------------- //

    virtual inline const char* type () const override
    {
        return "AccumulatorLoss";
    }


protected:

    virtual void Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;
//    virtual void Forward_gpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

    virtual void Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                               const vector<Blob<Dtype>*> &bottom) override;
//    virtual void Backward_gpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
//                               const vector<Blob<Dtype>*> &bottom) override;

    /**
     * @brief Builds the accumulator from the given labels
     * @param labels Blob with the labels of shape batch x num_bbs x 5
     */
    void _buildAccumulator (const Blob<Dtype> *labels);

    /**
     * @brief Applies a random mask to diffs to ensure the 1:1 ratio of positive and negative samples
     * @return Number of active samples
     */
    Dtype _applyMask();


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // Hough map - accumulator of object centers (shape of the blob is the same as bottom[0])
    Blob<Dtype> _accumulator;
    // Diff of the accumulator and the net output - saved for backpropagation
    Blob<Dtype> _diff;
    // Random number generator
    shared_ptr<Caffe::RNG> _rng;

};


}  // namespace caffe


#endif  // CAFFE_ACCUMULATOR_LOSS_LAYER_HPP_
