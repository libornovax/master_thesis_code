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

    virtual void _loadBBTXTFile ();

    virtual void _shuffleImages ();


    // ---------------------------------------  PROTECTED MEMBERS  --------------------------------------- //
    // List of image paths and 2D bounding box annotations in the form of a blob
    std::vector<std::pair<std::string, std::shared_ptr<Blob<Dtype>>>> _images;
    int _i_global;
    // Random number generator
    shared_ptr<Caffe::RNG> _rng;

};


}  // namespace caffe


#endif  // CAFFE_BBTXT_DATA_LAYER_HPP_
