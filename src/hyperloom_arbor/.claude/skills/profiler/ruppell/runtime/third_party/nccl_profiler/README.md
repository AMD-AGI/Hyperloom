# NCCL's profiler-plugin headers

The headers of NCCL's profiler-plugin interface, unchanged, from NCCL `v2.29.2-1`
(`ext-profiler/example/nccl/`), under NCCL's license (`LICENSE.txt`). ruppell's NCCL profiler
plugin (`runtime/src/nccl_profiler.cpp`, roadmap-v2 M8) is built against them. NCCL loads a
plugin by the newest interface version it exports, and the plugin exports `ncclProfiler_v5`.
