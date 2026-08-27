# Token Optimization with RTK

## What is RTK?
RTK (Rust Token Killer) is a CLI proxy that filters verbose command output
before it reaches the LLM context window. This saves 60-90% of tokens on
common development operations.

Source: https://github.com/rtk-ai/rtk

## How It's Integrated
The GPU toolchain automatically routes commands through RTK when available:

- **ninja/cmake build output**: ~80-90% savings (progress bars, timestamps stripped)
- **git operations**: 59-80% savings (compact status, log, diff)
- **general commands**: Smart filtering based on command type

Commands whose output we parse programmatically (rocprofv3 CSV, llvm-objdump)
bypass RTK to preserve raw format.

## Impact on Kernel Development
A typical CK build produces 500+ lines of output. RTK reduces this to
essential information (errors, warnings, success/failure), saving ~400 tokens
per build. Over 100 iterations of autonomous optimization, this saves
~40,000 tokens — roughly $1.20 at Opus pricing.

## Verification
```bash
rtk --version          # Check installation
rtk gain               # View token savings statistics
rtk gain --history     # View per-command savings history
```

## When NOT to Use RTK
- rocprofv3 output (we parse CSV for counter values)
- llvm-objdump (we parse register metadata)
- readelf (we parse ELF notes)

These are automatically excluded by the smart_wrap() function.
