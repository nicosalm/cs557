#pragma once

#include <chrono>
#include <cstring>
#include <iostream>

struct Timer
{
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;

    time_point_t mStartTime;
    time_point_t mStopTime;

    void Start()
    {
        mStartTime = clock_t::now();
    }

    void Stop(const std::string& msg)
    {
        mStopTime = clock_t::now();
        std::chrono::duration<double, std::milli> elapsedTime = mStopTime - mStartTime;
        std::cout << "[" << msg << elapsedTime.count() << "ms]" << std::endl;
    }

    /* why is the first iteration so much more ms?
     *  1. instruction cache? for smaller kernels it could be an issue
     *  2. lazy os? what does `new` do -- with mmap, page fault the first time (not yet materialized in physical ram, page fault into os and go do that stuff)
     *  3. tlb: cache that caches translations between virtual mem and physical mem;
     *          pages can be as small as 4kb, consider 2GB -- certainly will not have preloaded TLB with the addrs of the pages, need to discover
     *      - tlb replacement policy?
     *  4. memory access costs and latency (-most important-):
     *      - how latency is being hidden (big difference between gpus and cpus)
     *          gpus: massive hyperthreading (parallel exec strings alive at every point in time, round robining into execution)
     *              - by spacing out any two reqs from a stream of execution, can absorb latency by spacing out difference when you request something &
     *                when you truly need it
     *          cpus:  prefetching (how am i accessing memory over here? straight line? hmm... useful info!)
     *              - memory manager in cpu can pick up patterns and is preemptive
     */
};

// &u[i][j] = &u[0][0] + j + (i * Y_dim)
// cache line: usually 64 bytes (CPU)
