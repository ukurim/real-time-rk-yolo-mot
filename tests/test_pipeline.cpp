#include "aerial/pipeline.hpp"
#include <cassert>
#include <future>
#include <iostream>

using namespace aerial;
DetectionPacket packet(std::uint64_t id) { DetectionPacket p; p.frame.id = id; return p; }
int main() {
    FrameSlot<int> latest(true);
    latest.put(1); latest.put(2); latest.put(3);
    int item = 0;
    assert(latest.dropped() == 2 && latest.get(item) && item == 3);
    latest.close(); assert(!latest.get(item)); assert(!latest.put(4));

    CompletionQueue realtime(true, 3);
    DetectionPacket p;
    realtime.put(packet(5)); assert(realtime.get(p) && p.frame.id == 5);
    realtime.put(packet(2)); realtime.put(packet(4)); // late results cannot overwrite an already-consumed result
    realtime.put(packet(8)); realtime.put(packet(7));
    assert(realtime.get(p) && p.frame.id == 8);
    assert(realtime.dropped() == 3);
    realtime.close(); assert(!realtime.get(p));

    CompletionQueue ordered(false, 2);
    ordered.put(packet(1)); ordered.put(packet(2));
    // Reorder map is full but missing frame 0 must still be admitted.
    auto earliest = std::async(std::launch::async, [&]{ return ordered.put(packet(0)); });
    assert(earliest.wait_for(std::chrono::seconds(1)) == std::future_status::ready && earliest.get());
    for (unsigned i = 0; i < 3; ++i) { assert(ordered.get(p)); assert(p.frame.id == i); }
    ordered.close(); assert(!ordered.get(p));

    FrameSlot<int> blocking(false);
    blocking.put(1);
    auto producer = std::async(std::launch::async, [&]{ return blocking.put(2); });
    assert(producer.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    blocking.close();
    assert(producer.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !producer.get());
    FrameSlot<int> cancellable(false);
    cancellable.put(1);
    std::atomic<bool> cancel{false};
    auto cancelled = std::async(std::launch::async, [&]{ return cancellable.put(2, &cancel); });
    assert(cancelled.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    cancel = true;
    assert(cancelled.wait_for(std::chrono::seconds(1)) == std::future_status::ready && !cancelled.get());
    std::cout << "Queue replacement, ordering, bounded reorder and shutdown tests passed\n";
}
