//
// Libor Novak
// 03/13/2017
//

#ifndef CAFFE_BBTXT_LOSS_LAYER_HPP_
#define CAFFE_BBTXT_LOSS_LAYER_HPP_

#include <vector>
#include <atomic>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/internal_threadpool.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_counter.hpp"


namespace caffe {


/**
 * @brief The BBTXTLossLayer class
 *
 * The loss is composed of two different losses - Accumulator loss and regression loss from bounding box
 * coordinates. This loss supports only one accumulator! I we want multiple accumulators in the network, each
 * accumulator has to have its own bbtxt loss layer
 */
template <typename Dtype>
class BBTXTLossLayer : public LossLayer<Dtype>, public InternalThreadpool
{
public:

    explicit BBTXTLossLayer (const LayerParameter &param);
    virtual ~BBTXTLossLayer ();

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

    virtual void InternalThreadpoolEntry (int t) override;

    /**
     * @brief Computes the bounds of bounding box sizes for the accumulator
     */
    virtual void _computeSizeBounds ();

    /**
     * @brief Builds the accumulators from the given labels
     * @param b Id of image in the batch
     */
    virtual void _buildAccumulator (int b);

    /**
     * @brief Removes diffs of coordinates of negative samples - we do not want to include them in the loss
     * @param b Id of image in the batch
     * @return Number of removed coordinates
     */
    virtual int _removeNegativeCoordinateDiff (int b);

    /**
     * @brief Computes the squared error of the probability channel
     * @param b Id of image in the batch
     */
    virtual void _computeProbabilityLoss (int b);

    /**
     * @brief Weights the diffs in order to even out the impact of positive and negative samples on the gradient
     * @param b Id of image in the batch
     */
    virtual void _applyDiffWeights (int b);

    /**
     * @brief Renders a circle with coordinates (xmin, ymin, xmax or ymax)
     * It has to be rendered specifically because there is a different value in each pixel of the accumulator
     * @param acc Accumulator of the coordinate to be rendered
     * @param x Coordinate of the center in the original image coordinates (not scaled)
     * @param y Coordinate of the center in the original image coordinates (not scaled)
     * @param value Not scaled coordinate to be rendered
     * @param channel 1, 2, 3 or 4 - determined which coordinate is being rendered
     */
    virtual void _renderCoordinateCircle (cv::Mat &acc, int x, int y, Dtype value, int channel);


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // Hough map - accumulator of object centers (shape of the blob is the same as bottom[0])
    std::shared_ptr<Blob<Dtype>> _accumulator;
    // Scale (downsampling) of the accumulator
    int _scale;
    // Each accumulator only includes some bounding boxes - given the size of the boundig box. The _bb_bounds
    // contain bounds on the bounding box sizes, which are going to be included in this accumulator
    std::pair<float,float> _bb_bounds;
    float _ideal_size;

    // Blob with the labels of shape batch x num_bbs x 5
    const Blob<Dtype>* _labels;
    const Blob<Dtype>* _bottom;

    std::shared_ptr<Blob<Dtype>> _diff;

    // Queue of indices of images to be processed
    BlockingQueue<int> _b_queue;
    BlockingCounter _num_processed;

    std::atomic<Dtype> _loss_prob;
    std::atomic<Dtype> _loss_prob_pos;
    std::atomic<Dtype> _loss_prob_neg;
    std::atomic<Dtype> _loss_coord;

};


}  // namespace caffe


#endif  // CAFFE_BBTXT_LOSS_LAYER_HPP_
