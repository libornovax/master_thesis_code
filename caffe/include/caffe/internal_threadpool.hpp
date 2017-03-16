//
// Libor Novak
// 03/16/2017
//

#ifndef CAFFE_INTERNAL_THREADPOOL_HPP_
#define CAFFE_INTERNAL_THREADPOOL_HPP_

#include "caffe/common.hpp"


/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }


namespace caffe {

/**
 * Inspired by the InternalThread class.
 *
 * The child class will acquire the ability to run a multiple threads, by reimplementing the virtual
 * function InternalThreadEntry.
 */
class InternalThreadpool {
public:

    InternalThreadpool (int num_threads);
    virtual ~InternalThreadpool ();


    /**
     * @brief Starts code execution on all threads
     */
    void StartInternalThreadpool ();

    /**
     * @brief Will not return until the internal thread has exited
     */
    void StopInternalThreadpool ();

    /**
     * @brief Checks if the thread is running
     * @param t Thread id
     */
    bool is_started (int t) const;


protected:

    /**
     * @brief Implement this method in your subclass with the code you want your thread to run
     * @ t Thread id
     */
    virtual void InternalThreadEntry (int t) = 0;

    /**
     * @brief Should be tested when running loops to exit when requested
     * @param t Thread id
     */
    bool must_stop (int t);


private:

    void entry (int t, int device, Caffe::Brew mode, int rand_seed, int solver_count, int solver_rank,
                bool multiprocess);


    // ----------------------------------------  PRIVATE MEMBERS  ---------------------------------------- //
    // List of running threads
    std::vector<shared_ptr<boost::thread>> _threads;

};


}  // namespace caffe


#endif  // CAFFE_INTERNAL_THREADPOOL_HPP_
