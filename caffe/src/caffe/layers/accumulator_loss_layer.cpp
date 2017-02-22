#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/accumulator_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
AccumulatorLossLayer<Dtype>::AccumulatorLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param),
      _accumulator(),
      _diff()
{
}


template <typename Dtype>
void AccumulatorLossLayer<Dtype>::Reshape (const vector<Blob<Dtype>*> &bottom,
                                           const vector<Blob<Dtype>*> &top)
{
    LossLayer<Dtype>::Reshape(bottom, top);

    this->_accumulator.ReshapeLike(*bottom[0]);
    this->_diff.ReshapeLike(*bottom[0]);
}


template <typename Dtype>
void AccumulatorLossLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom,
                                               const vector<Blob<Dtype>*> &top)
{
    // Create the accumulator from the labels
    this->_buildAccumulator(bottom[1]);

    // Now it is a simple squared Euclidean distance of the net output and the accumulator
    int count = bottom[0]->count();
    caffe_sub(count, bottom[0]->cpu_data(), this->_accumulator.cpu_data(), this->_diff.mutable_cpu_data());
    Dtype dot = caffe_cpu_dot(count, _diff.cpu_data(), _diff.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2); // Divide by the number of images - per image loss

    // Write the loss to the output blob
    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void AccumulatorLossLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top,
                                                const vector<bool> &propagate_down,
                                                const vector<Blob<Dtype>*> &bottom)
{
    if (propagate_down[0])
    {
        const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
        caffe_cpu_axpby(bottom[0]->count(), alpha, this->_diff.cpu_data(), Dtype(0),
                        bottom[0]->mutable_cpu_diff());
    }
}



// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

template <typename Dtype>
void AccumulatorLossLayer<Dtype>::_buildAccumulator (const Blob<Dtype> *labels)
{
    // This is because of the cv::Mat type - I don't want to program other types now except CV_32FC1
    CHECK((std::is_same<Dtype,float>::value)) << "AccumulatorLossLayer supports only float!";

    const int radius = 1;
    const double scale = 0.25;

    const int height = this->_accumulator.shape(2);
    const int width  = this->_accumulator.shape(3);

    Dtype *accumulator_data = this->_accumulator.mutable_cpu_data();

    for (int b = 0; b < labels->shape(0); ++b)
    {
        // Create a cv::Mat wrapper for the accumulator - this way we can now use OpenCV drawing functions
        // to create circles in the accumulator
        cv::Mat acc(height, width, CV_32FC1, accumulator_data + this->_accumulator.offset(b));
        acc.setTo(cv::Scalar(0));

        // Draw circles in the center of the bounding boxes
        for (int i = 0; i < labels->shape(1); ++i)
        {
            const Dtype *data = labels->cpu_data() + labels->offset(b, i);
            if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 0 && data[4] == 0) break;

            cv::circle(acc, cv::Point(scale*(data[1]+data[3])/2, scale*(data[2]+data[4])/2), radius,
                       cv::Scalar(1), -1);
        }


    }
}



// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(AccumulatorLossLayer);
#endif

INSTANTIATE_CLASS(AccumulatorLossLayer);
REGISTER_LAYER_CLASS(AccumulatorLoss);


}  // namespace caffe
