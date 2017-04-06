//
// Libor Novak
// 04/06/2017
//

#ifndef CAFFE_BB3TXT_SCALING_LAYER_HPP_
#define CAFFE_BB3TXT_SCALING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"


namespace caffe {


/**
 * @brief BB3TXTBBLayer
 * Scales the coordinates on the output of the accumulators back to pixel coordinates (in the network
 * the coordinates are normalized to have approximately 0 mean and unit variance) and computes the absolute
 * position of the bounding boxes in the image (with respect to 0,0 of the original image)
 */
template <typename Dtype>
class BB3TXTBBLayer : public NeuronLayer<Dtype>
{
public:

    explicit BB3TXTBBLayer (const LayerParameter &param);


    // -----------------------------------------  INLINE METHODS  ---------------------------------------- //

    virtual inline const char* type () const override
    {
        return "BBTXTScaling";
    }


protected:

    virtual void Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;
//    virtual void Forward_gpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) override;

    virtual void Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                               const vector<Blob<Dtype>*> &bottom) override;
//    virtual void Backward_gpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
//                               const vector<Blob<Dtype>*> &bottom) override;


};


}  // namespace caffe

#endif  // CAFFE_BB3TXT_SCALING_LAYER_HPP_
