#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_threadpool.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


InternalThreadpool::InternalThreadpool (int num_threads)
{
    for (int t = 0; t < num_threads; ++t) this->_threads.emplace_back();
}


InternalThreadpool::~InternalThreadpool ()
{
    StopInternalThreadpool();
}


bool InternalThreadpool::is_started (int t) const
{
    return this->_threads[t] && this->_threads[t]->joinable();
}


bool InternalThreadpool::must_stop (int t)
{
    return this->_threads[t] && this->_threads[t]->interruption_requested();
}


void InternalThreadpool::StartInternalThreadpool ()
{
    for (int t = 0; t < this->_threads.size(); ++t)
    {
        CHECK(!is_started(t)) << "Threads should persist and not be restarted.";

        int device = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDevice(&device));
#endif
        Caffe::Brew mode = Caffe::mode();
        int rand_seed = caffe_rng_rand();
        int solver_count = Caffe::solver_count();
        int solver_rank = Caffe::solver_rank();
        bool multiprocess = Caffe::multiprocess();

        try {
            this->_threads[t].reset(new boost::thread(&InternalThreadpool::entry, this, t, device, mode,
                                                      rand_seed, solver_count, solver_rank, multiprocess));
        } catch (std::exception& e) {
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }
}


void InternalThreadpool::entry (int t, int device, Caffe::Brew mode, int rand_seed, int solver_count,
                                int solver_rank, bool multiprocess)
{
#ifndef CPU_ONLY
    CUDA_CHECK(cudaSetDevice(device));
#endif
    Caffe::set_mode(mode);
    Caffe::set_random_seed(rand_seed);
    Caffe::set_solver_count(solver_count);
    Caffe::set_solver_rank(solver_rank);
    Caffe::set_multiprocess(multiprocess);

    InternalThreadEntry(t);
}


void InternalThreadpool::StopInternalThreadpool ()
{
    for (int t = 0; t < this->_threads.size(); ++t)
    {
        if (this->is_started(t))
        {
            this->_threads[t]->interrupt();

            try {
                this->_threads[t]->join();
            } catch (boost::thread_interrupted&) {
            } catch (std::exception& e) {
                LOG(FATAL) << "Thread exception: " << e.what();
            }
        }
    }
}


}  // namespace caffe
