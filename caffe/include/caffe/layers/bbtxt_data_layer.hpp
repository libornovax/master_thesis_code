//
// Libor Novak
// 02/21/2017
//

#ifndef CAFFE_BBTXT_DATA_LAYER_HPP_
#define CAFFE_BBTXT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <mutex>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_threadpool.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_counter.hpp"


namespace caffe {


/**
 * @brief Stuct for storing indices of images and bounding boxes
 */
template <typename Dtype>
struct SelectedBB
{
    std::string filename;
    std::shared_ptr<Blob<Dtype>> label;
    int bb_id;
};


/**
 * @brief The BBTXTDataLayer class
 *
 * The BBTXTDataLayer loads a BBTXT file with 2D bounding boxes and runs learning on the images specified in
 * the paths in the given file.
 */
template <typename Dtype>
class BBTXTDataLayer : public BasePrefetchingDataLayer<Dtype>, public InternalThreadpool
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

    virtual void InternalThreadpoolEntry (int t) override;

    /**
     * @brief Loads the whole input file in the memory and creates label blobs
     */
    virtual void _loadBBTXTFile ();

    /**
     * @brief Shuffle images in the dataset
     */
    virtual void _shuffleBoundingBoxes ();

    /**
     * @brief Crops a window from the given image around the given bb and resamples it to the network input blob
     * @param cv_img Image to be cropped from
     * @param b Id of image in the batch (for output blobs)
     * @param bb_id Id of selected bounding box in the image label
     */
    virtual void _cropAndTransform (const cv::Mat &cv_img, int b, int bb_id);

    /**
     * @brief Subroutine of the previous method. Performs only the cropping part with label adjustment
     * @param cv_img Image to be cropped from
     * @param cv_img_cropped_out Output cropped image
     * @param b Id of image in the batch (for output blobs)
     * @param bb_id Id of selected bounding box in the image label
     */
    virtual void _cropBBFromImage (const cv::Mat &cv_img, cv::Mat &cv_img_cropped_out, int b, int bb_id);

    /**
     * @brief Converts values to 0 mean, unit variance and adds some hue, exposure,... adjustments
     * @param cv_img_cropped Already cropped image
     * @param b Id of image in the batch (for output blobs)
     */
    virtual void _applyPixelTransformationsAndCopyOut (const cv::Mat &cv_img_cropped, int b);

    /**
     * @brief Randomly flips the image during training
     * @param cv_img_cropped Already cropped and rescaled image
     * @param b Id of image in the batch (for output blobs)
     */
    virtual void _flipImage(cv::Mat &cv_img_cropped, int b);

    /**
     * @brief Thread safe selecting of image filename and id of bounding box in that image
     * Operates on _i_global and _bb_id_global
     * @param b Id of image in the batch (for output blobs)
     * @return
     */
    virtual SelectedBB<Dtype> _getImageAndBB ();


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // List of image paths and 2D bounding box annotations in the form of a blob
    std::vector<std::pair<std::string, std::shared_ptr<Blob<Dtype>>>> _images;
    // Vector with indices of all bounding boxes in the dataset
    std::vector<std::pair<int, int>> _indices;
    // Indices for loading images
    int _i_global;
    // Random number generator
    shared_ptr<Caffe::RNG> _rng;

    // Queue of indices of images to be processed
    BlockingQueue<int> _b_queue;
    BlockingCounter _num_processed;
    // Mutex for access to _i_global and _bb_id_global
    mutable std::mutex _i_global_mtx;

};


}  // namespace caffe


#endif  // CAFFE_BBTXT_DATA_LAYER_HPP_
