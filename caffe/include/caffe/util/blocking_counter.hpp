//
// Libor Novak
// 03/16/2017
//

#ifndef CAFFE_UTIL_BLOCKING_COUNTER_HPP_
#define CAFFE_UTIL_BLOCKING_COUNTER_HPP_

#include <string>

namespace caffe {


/**
 * @brief Similar logic to blocking queue, but this class is made to contain a counter for which we have to
 * wait to count to a certain value. I.e. wait for some number of elements to be processed
 */
class BlockingCounter {
public:

    explicit BlockingCounter ();


    void reset ();

    void increase ();
    void decrease ();

    int getCount ();

    /**
     * @brief Blocks execution of the thread that called it until the counter counts to the given number
     * @param count
     */
    void waitToCount (int count);


protected:

    /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
    class sync;

    int _counter;
    std::shared_ptr<sync> _sync;


    DISABLE_COPY_AND_ASSIGN(BlockingCounter);
};

}  // namespace caffe

#endif
