# Choose a Windows release

| Channel | vLLM | Installer Python | Torch / CUDA | Release |
|---|---|---|---|---|
| **Latest** | 0.33.0 | 3.13.14 | 2.14.0+cu130 / 13.0 | [Python 3.13 / cu130](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130) |
| Python 3.14 build | 0.33.0 | 3.14.2 | 2.14.0+cu132 / 13.2 | [Python 3.14 / cu132](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu132-py314) |
| Older release | 0.29.0 | 3.14.2 | 2.13.0+cu132 / 13.2 | [Python 3.14 / cu132](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu132-py314) |
| Older prerelease | 0.29.0 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Python 3.13 / cu130](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1) |
| Older release | 0.27.1 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Python 3.13 / cu130](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.27.1-win-cu130) |

**Most users want Latest.** The Python 3.14 build is the same vLLM version for
people who need Python 3.14; it uses **CPU TorchAudio**, so GPU audio processing
and audio-model serving were not validated (vLLM model inference still uses
CUDA). Older releases stay available for anyone who needs to stay on them;
see [all releases](https://github.com/aivrar/vllm-windows-build/releases).

The 0.29.0 Python 3.14 release includes the corrected `launch.bat` in its
source ZIP. The earlier `rc1` source ZIP does not; existing `rc1` installs can
replace their launcher with the [corrected asset](https://github.com/aivrar/vllm-windows-build/releases/download/v0.29.0-win-cu132-py314-rc1/launch.bat).

## Install a prebuilt wheel

Download **Source code (zip)** from your chosen release, extract into a new
directory, and run `install.bat`, then `launch.bat`. The installer downloads
Python and the matching prebuilt wheels; **no compilation is required**.
Keep existing installations separate and use a fresh filesystem KV-cache
directory when switching versions.

A default `git clone` checks out `master`, whose installer provides the Latest
release (0.33.0, Python 3.13). Git itself does not download a Python runtime.
For another release, clone its tag into a separate directory:

```bat
git clone --branch v0.33.0-win-cu132-py314 https://github.com/aivrar/vllm-windows-build.git vllm-py314-cu132
```

Release tags identify fixed installer/source snapshots. The Python 3.14 build
is maintained on `release/v0.33.0-win-cu132-py314`; the older 0.29.0 variants are on
`prerelease/v0.29.0-win-cu130` and `release/v0.29.0-win-cu132-py314`.

## Compile your own wheel

`install.bat` installs prebuilt wheels; it is not the source-build workflow.
Use the build record for the selected version, including its Python/CUDA/Torch
pins, upstream source revision, patches and vendor revisions:

- [0.33.0 Python 3.13 / cu130 source/build record](v0.33.0-build-candidate.md).
- [0.33.0 Python 3.14 / cu132 build record](https://github.com/aivrar/vllm-windows-build/blob/v0.33.0-win-cu132-py314/docs/v0.33.0-cu132-py314.md).
- [0.29.0 Python 3.14 / cu132 source deltas and build pins](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu132-py314/docs/v0.29.0-cu132-py314.md#exact-source-and-build-changes).
- [0.29.0 Python 3.13 / cu130 source/build record](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu130-rc1/docs/v0.29.0-build-candidate.md).
- [0.27.1 source/build record](v0.27.1-build-candidate.md).

The Python patch version in a historical build record can differ from the
portable installer's patch version; the table above describes the installers.
Historical root `build.bat` targets 0.25.1; running it is not a way to build
0.29.0 or later. Maintainer workspace paths in build records describe the
recorded environment; substitute your own paths when reproducing it.

## Validation and limitations

Both 0.33.0 wheels passed targeted RTX 3090 checks: real-model serving with
compilation and CUDA graphs, six local KV formats, the issue #16 FP8 Marlin
regression, a CPU KV-offload restore, and a clean-folder install and launch.
They were not tested on Blackwell hardware, and **native Blackwell FP4 is not
added**. Upstream removed the Rust tool-parser extension in 0.33.0.

Both 0.29.0 wheels passed targeted RTX 3090 checks and include the issue #16
Marlin fallback fixes. An RTX 5090 tester also confirmed the 0.29.0 Python 3.14
build with one specific Qwen3.8-27B INT4 model at 128K context, FP8 KV and MTP.
Exact scope is recorded on each release page. No speed advantage is claimed
for Python 3.14/cu132 or for MTP. The 0.27.1 Blackwell workaround remains
documented for users staying on 0.27.1.
