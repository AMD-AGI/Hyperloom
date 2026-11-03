// gray-parrot: ruppell's hardware counters on ROCm (docs/gray-parrot/README.md).
//
// The collection lives in gray_parrot.cpp, compiled into the tool library beside tool.cpp,
// which calls only these hooks. With RUPPELL_COUNTERS and RUPPELL_DEVICE_COUNTERS unset,
// configure() creates nothing (no context, buffer, thread or file) and every other hook
// returns at once.
#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <cstdint>
#include <string>

namespace gray_parrot
{
// tool_init, before ruppell's own contexts start: build the counter configurations and
// configure the counting services. Services can be configured only at init, so counters cannot
// be switched on later in a run.
void configure(const std::string& out_dir, rocprofiler_callback_thread_t cb_thread);
// tool_init, after the contexts start: the device sampler's thread.
void start();
// the code-object callback: a kernel symbol registered (RUPPELL_COUNTER_KERNELS decides here)
void kernel_symbol(uint64_t kernel_id, const char* name);
void flush();
void fini();
// a forked child: drop the parent's streams, record nothing
void forget_in_child();
// meta.json's "counters": "off", or what was configured and collected
std::string meta_json();
}  // namespace gray_parrot

// tool.cpp's, for gray-parrot: whether kernel dispatch records are on, and the SDK's clock
namespace ruppell_tool
{
int      detail_state();
uint64_t now();
}  // namespace ruppell_tool
