//
// Libor Novak
// 02/21/2017
//

#ifndef CAFFE_BBTXT_DATA_LAYER_HPP_
#define CAFFE_BBTXT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


/**
 * @brief The BBTXTDataLayer class
 *
 * The BBTXTDataLayer loads a BBTXT file with 2D bounding boxes and runs learning on the images specified in
 * the paths in the given file.
 */
template <typename Dtype>
class BBTXTDataLayer : public BasePrefetchingDataLayer<Dtype>
{
public:

    explicit BBTXTDataLayer (const LayerParameter &param);
    virtual ~BBTXTDataLayer ();

    /**
     * @brief Load the BBTXT file and check if images exist
     */
    virtual void DataLayerSetUp (const vector<Blob<Dtype>*> &bottom,
                                 const vector<Blob<Dtype>*> &top) override;


    // -----------------------------------------  INLINE METHODS  ---------------------------------------- //

    virtual inline const char* type () const override
    {
        return "BBTXTData";
    }

    virtual inline int ExactNumBottomBlobs () const override
    {
        return 0;
    }

    virtual inline int ExactNumTopBlobs () const override
    {
        // Data and labels
        return 2;
    }


protected:

    virtual void load_batch (Batch<Dtype> *batch) override;

    /**
     * @brief Loads the whole input file in the memory and creates label blobs
     */
    virtual void _loadBBTXTFile ();

    /**
     * @brief Shuffle images in the dataset
     */
    virtual void _shuffleImages ();

    /**
     * @brief Crops a window from the given image and resamples it to the network input blob
     * @param cv_img Image to be cropped from
     * @param transformed_image Output blob (input blob of the network)
     * @param transformed_label Annotation of the image - will be transformed according to crop
     */
    virtual void _cropAndTransform (const cv::Mat &cv_img, Blob<Dtype> &transformed_image,
                                    Blob<Dtype> &transformed_label);

    /**
     * @brief Subroutine of the previous method. Performs only the cropping part with label adjustment
     * @param cv_img Image to be cropped from
     * @param cv_img_cropped_out Output cropped image
     * @param transformed_label Annotation of the image - will be transformed according to crop
     * @param bb_id Id of the bounding box that should be used to specify the crop
     */
    virtual void _cropBBFromImage (const cv::Mat &cv_img, cv::Mat &cv_img_cropped_out,
                                   Blob<Dtype> &transformed_label, int bb_id);

    /**
     * @brief Converts values to 0 mean, unit variance and adds some hue, exposure,... adjustments
     * @param cv_img_cropped Already cropped image
     * @param transformed_image Input to the network to which the transformed image will be copied
     */
    virtual void _applyPixelTransformationsAndCopyOut (const cv::Mat &cv_img_cropped,
                                                       Blob<Dtype> &transformed_image);


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // List of image paths and 2D bounding box annotations in the form of a blob
    std::vector<std::pair<std::string, std::shared_ptr<Blob<Dtype>>>> _images;
    // Indices for loading images
    int _i_global;
    int _bb_id_global;
    // Random number generator
    shared_ptr<Caffe::RNG> _rng;

};


}  // namespace caffe


#endif  // CAFFE_BBTXT_DATA_LAYER_HPP_
