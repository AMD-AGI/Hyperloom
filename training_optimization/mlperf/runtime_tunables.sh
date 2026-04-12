#!/bin/bash

echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo modprobe amdgpu
rocm_agent_enumerator
sudo cpupower idle-set -d 2
sudo cpupower frequency-set -g performance
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag