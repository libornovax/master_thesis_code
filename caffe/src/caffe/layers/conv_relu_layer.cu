#include <vector>

#include "caffe/layers/conv_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ConvolutionReluLayer<Dtype>::Forward_gpu (const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i)
    {
        // Convolution
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n)
        {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                   top_data + n * this->top_dim_);
            if (this->bias_term_)
            {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }

        // Relu
        const int count = top[i]->count();
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
        ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data, top_data, negative_slope);
        CUDA_POST_KERNEL_CHECK;
    }
}

template <typename Dtype>
void ConvolutionReluLayer<Dtype>::Backward_gpu (const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    CHECK(false) << "ConvolutionReluLayer implements only the forward pass!";
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionReluLayer);

}  // namespace caffe
