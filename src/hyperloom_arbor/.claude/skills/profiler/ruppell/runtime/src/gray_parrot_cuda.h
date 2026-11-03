// gray-parrot: ruppell's hardware counters on CUDA (docs/gray-parrot/README.md, §1.4's note).
//
// The collection lives in gray_parrot_cuda.cpp, compiled into the CUDA tool library beside
// tool_cuda.cpp, which calls only these hooks. With RUPPELL_COUNTERS and RUPPELL_DEVICE_COUNTERS
// unset, configure() reads two variables and nothing else happens: no CUPTI profiling call, no
// file, and every other hook returns at once.
#pragma once

#include <cstdint>
#include <string>

namespace gray_parrot_cuda
{
// tool_init: the switches, the groups asked for. No CUPTI profiling call (the Range Profiler
// needs a context, which tool_init runs before).
void configure(const std::string& out_dir);
// the groups a counter step can count, or -1 without counters (unset or refused)
int groups();
// a counter step begins: the Range Profiler is set up at the first (on the calling thread's
// context), then a user range starts. Returns the group counted, -1 without counters, -2 when
// CUPTI did not start (counted in meta.json), -3 for a group out of range.
int start(int group, uint32_t step, uint64_t now);
// the counter step ends (the caller has drained the device): the range stops, its values go to
// device_counters.bin. Returns 0, or -2 when CUPTI did not stop or decode (counters are then
// refused for the rest of the run).
int stop(uint64_t now);
bool counting();
// a context was created on `device` (CUDA's ordinal): with RUPPELL_DEVICE_COUNTERS, the device's PM
// sampling session starts on a thread of its own, once per device
void context_created(int device);
void flush();
void fini();
// fini also stops the devices' PM sampling and writes their last samples
// meta.json's "counters": "off", or what was configured and collected
std::string meta_json();
}  // namespace gray_parrot_cuda
