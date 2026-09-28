#include "aerial/output.hpp"
#include <cassert>
#include <future>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    // A named FIFO with no reader must fail at open rather than hanging before
    // the publisher thread or cancellation mechanism has even been created.
    char directory[] = "/tmp/aerial-json-fifo-XXXXXX";
    assert(mkdtemp(directory));
    const std::string fifo = std::string(directory) + "/results";
    assert(mkfifo(fifo.c_str(), 0600) == 0);
    const auto started = std::chrono::steady_clock::now();
    bool rejected = false;
    try { aerial::JsonPublisher absent_reader(fifo, true); }
    catch (const std::runtime_error&) { rejected = true; }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    assert(unlink(fifo.c_str()) == 0 && rmdir(directory) == 0);
    assert(rejected && elapsed < std::chrono::seconds(1));

    // A real full pipe exercises I/O backpressure, not a mocked writer.
    int pipe_fds[2];
    assert(pipe(pipe_fds) == 0);
    const int saved_stdout = dup(STDOUT_FILENO);
    assert(saved_stdout >= 0 && dup2(pipe_fds[1], STDOUT_FILENO) >= 0);
    {
        aerial::JsonPublisher output("-", true);
        const std::string record = "{\"payload\":\"" + std::string(256 * 1024, 'x') + "\"}\n";
        assert(output.publish(record));
        // Let the publisher fill the OS pipe. New perception results must still
        // be accepted without waiting for the absent reader.
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        auto producer = std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i) if (!output.publish(record)) return false;
            return true;
        });
        assert(producer.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
        assert(producer.get() && output.dropped() > 0);
        output.finish();
        assert(output.failed()); // Explicit stalled-consumer error, bounded shutdown.
    }
    assert(dup2(saved_stdout, STDOUT_FILENO) >= 0);
    close(saved_stdout); close(pipe_fds[0]); close(pipe_fds[1]);
    std::cout << "Absent FIFO reader, slow live JSON consumer, and bounded shutdown tests passed\n";
}
