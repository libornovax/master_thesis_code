#include <algorithm>
#include <vector>

#include "caffe/layers/bb3txt_bb_layer.hpp"

namespace caffe {


template <typename Dtype>
BB3TXTBBLayer<Dtype>::BB3TXTBBLayer (const LayerParameter &param)
    : NeuronLayer<Dtype>(param)
{
    CHECK(param.has_bbtxt_bb_param()) << "BBTXTScalingParameter is mandatory!";
    CHECK(param.bbtxt_bb_param().has_ideal_size()) << "Ideal size is mandatory!";
    CHECK(param.bbtxt_bb_param().has_downsampling()) << "Downsampling is mandatory!";
}


template <typename Dtype>
void BB3TXTBBLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    const Dtype ideal_size   = this->layer_param_.bbtxt_bb_param().ideal_size();
    const Dtype downsampling = this->layer_param_.bbtxt_bb_param().downsampling();

    // For each image in the batch
    for (int b = 0; b < bottom[0]->shape(0); ++b)
    {
        // For each channel
        // fblx, fbly, fbrx, fbry, rblx, rbly, ftly
        for (int c = 1; c < 8; ++c)
        {
            const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(b, c);
            Dtype* top_data          = top[0]->mutable_cpu_data() + top[0]->offset(b, c);

            for (int i = 0; i < bottom[0]->shape(2); ++i)
            {
                for (int j = 0; j < bottom[0]->shape(3); ++j)
                {
                    // Convert the local normalized coordinate into a global unnormalized value in pixels
                    if (c == 1 || c == 3 || c == 5)
                    {
                        // fblx, fbrx, rblx
                        *top_data = int(downsampling*(j+0.5f)) + ideal_size * (*bottom_data - Dtype(0.5f));
                    }
                    else // (c == 2 || c == 4 || c == 6 || c == 7)
                    {
                        // fbly, fbry, rbly, ftly
                        *top_data = int(downsampling*(i+0.5f)) + ideal_size * (*bottom_data - Dtype(0.5f));
                    }

                    bottom_data++;
                    top_data++;
                }
            }
        }
    }
}


template <typename Dtype>
void BB3TXTBBLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype>*> &bottom)
{
    CHECK(false) << "BB3TXTBBLayer implements only the forward pass!";
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(BB3TXTBBLayer);
#endif

INSTANTIATE_CLASS(BB3TXTBBLayer);
REGISTER_LAYER_CLASS(BB3TXTBB);


}  // namespace caffe
