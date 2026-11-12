# Contributing

Start with the [release selection guide](docs/releases.md). `master` installs the Latest release (0.33.0, Python 3.13 / CUDA 13.0); the Python 3.14 build and older versions use separate release branches and Python/CUDA stacks. A source checkout does not download Python or build a wheel by itself.

For a bug, use the [bug report form](https://github.com/aivrar/vllm-windows-build/issues/new/choose). Include the exact release tag, GPU and driver, Windows/Python/PyTorch versions, model and quantization when relevant, launch command, and the first error. Remove tokens, passwords, and private data from logs. For a feature or documentation request, explain the desired behavior and a concrete use case.

For a pull request, describe the affected release variant and user-visible change. Run the targeted checks relevant to the change; the Windows audit workflow checks repository contracts and patch application. State which GPU behavior was actually tested. If an installer, wheel, or release artifact changes, update its URL, size, SHA-256, release notes, and matching documentation together. Do not change existing release tags or claim untested Blackwell FP4 or GPU audio support.

The [wiki](https://github.com/aivrar/vllm-windows-build/wiki) is a browsable guide. Repository Markdown is the source for release-specific build records; link to the exact tag when citing a release.
