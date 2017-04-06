#include <vector>

#include "caffe/layers/conv_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionReluLayer<Dtype>::compute_output_shape ()
{
    const int* kernel_shape_data = this->kernel_shape_.cpu_data();
    const int* stride_data = this->stride_.cpu_data();
    const int* pad_data = this->pad_.cpu_data();
    const int* dilation_data = this->dilation_.cpu_data();
    this->output_shape_.clear();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
        // i + 1 to skip channel axis
        const int input_dim = this->input_shape(i + 1);
        const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
        const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                / stride_data[i] + 1;
        this->output_shape_.push_back(output_dim);
    }
}

template <typename Dtype>
void ConvolutionReluLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    const Dtype* weight = this->blobs_[0]->cpu_data();

    for (int i = 0; i < bottom.size(); ++i)
    {
        // Convolution
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        for (int n = 0; n < this->num_; ++n)
        {
            this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                   top_data + n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->cpu_data();
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
            }
        }

        // Relu
        const int count = top[i]->count();
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
        for (int j = 0; j < count; ++j)
        {
            top_data[j] = std::max(top_data[j], Dtype(0)) + negative_slope * std::min(top_data[j], Dtype(0));
        }
    }
}

template <typename Dtype>
void ConvolutionReluLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    CHECK(false) << "ConvolutionReluLayer implements only the forward pass!";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionReluLayer);
#endif

INSTANTIATE_CLASS(ConvolutionReluLayer);

}  // namespace caffe
