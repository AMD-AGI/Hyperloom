@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
call "%~dp0..\..\tools\winrocm_env.bat" || exit /b 1
python -u "%~dp0build_ck_varlen.py" %*
