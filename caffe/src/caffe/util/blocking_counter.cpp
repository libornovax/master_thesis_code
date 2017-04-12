#include <boost/thread.hpp>
#include <string>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_counter.hpp"

namespace caffe {


class BlockingCounter::sync
{
public:
    mutable boost::mutex mtx;
    boost::condition_variable cond;
};



BlockingCounter::BlockingCounter()
    : _counter(0),
      _sync(new sync())

{
}


void BlockingCounter::reset ()
{
    boost::mutex::scoped_lock lock(this->_sync->mtx);
    this->_counter = 0;
}


void BlockingCounter::increase ()
{
    boost::mutex::scoped_lock lock(this->_sync->mtx);
    this->_counter++;
    this->_sync->cond.notify_one();
}


void BlockingCounter::decrease ()
{
    boost::mutex::scoped_lock lock(this->_sync->mtx);
    this->_counter--;
}


int BlockingCounter::getCount ()
{
    boost::mutex::scoped_lock lock(this->_sync->mtx);
    return this->_counter;
}


void BlockingCounter::waitToCount (int count)
{
    boost::mutex::scoped_lock lock(this->_sync->mtx);

    while (this->_counter < count)
    {
        this->_sync->cond.wait(lock);
    }
}


}  // namespace caffe
