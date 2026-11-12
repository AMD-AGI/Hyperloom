"""
vLLM OpenAI-compatible server launcher for Windows.

vLLM's built-in API server uses AsyncMPClient with ZMQ multiprocess,
which doesn't work on Windows (no fork, IPC sockets, etc.).

This launcher creates a lightweight FastAPI server wrapping the
synchronous LLM class (which uses InprocClient on Windows) and
serves the OpenAI-compatible chat/completions endpoint with SSE streaming.

Supports OpenAI function calling format: pass 'tools' in the request body
and tool calls are returned in the standard tool_calls response format.
Models with native tool calling templates (Qwen2.5, Llama 3.x, etc.)
get tools injected via their chat template. Other models get tool
descriptions injected into the system prompt with text-based parsing.
"""

from __future__ import annotations

import sys
import types
import os
import asyncio
import argparse
import json
import re
import time
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from engine_dispatcher import EngineDispatcher, EngineDispatchFailure

# Keep CUDA device numbering consistent with nvidia-smi on Windows.  This is
# especially important when --gpu-id is used on systems with mixed GPUs.
if sys.platform == "win32":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Stub uvloop on Windows before any vLLM imports
if sys.platform == "win32":
    uvloop = types.ModuleType("uvloop")
    uvloop.run = asyncio.run
    sys.modules["uvloop"] = uvloop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vllm_launcher")

DEFAULT_GPU_MEMORY_UTILIZATION = 0.6
DEFAULT_MAX_NUM_SEQS = 64
TURING_GPU_MEMORY_UTILIZATION = 0.89
TURING_MAX_NUM_SEQS = 1
TURING_MAX_NUM_BATCHED_TOKENS = 2048
GGUF_UNSUPPORTED_MESSAGE = (
    "Direct GGUF files are not supported by this Windows launcher. "
    "Use a Hugging Face-format model directory or ID containing "
    "config.json and safetensors/AWQ/GPTQ weights."
)


def apply_runtime_defaults(args: argparse.Namespace) -> None:
    """Apply general or Turing-specific defaults while preserving overrides."""

    if args.turing_compat:
        args.attention_backend = args.attention_backend or "TRITON_ATTN"
        args.kv_cache_dtype = args.kv_cache_dtype or "float16"
        args.block_size = args.block_size or 32
        args.enforce_eager = True
        if args.gpu_memory_utilization is None:
            args.gpu_memory_utilization = TURING_GPU_MEMORY_UTILIZATION
        if args.max_num_seqs is None:
            args.max_num_seqs = TURING_MAX_NUM_SEQS
        if args.max_num_batched_tokens is None:
            args.max_num_batched_tokens = TURING_MAX_NUM_BATCHED_TOKENS
        logger.info(
            "Turing compatibility profile enabled: backend=%s, kv_cache_dtype=%s, "
            "block_size=%s, enforce_eager=True, gpu_memory_utilization=%s, "
            "max_num_seqs=%s, max_num_batched_tokens=%s",
            args.attention_backend,
            args.kv_cache_dtype,
            args.block_size,
            args.gpu_memory_utilization,
            args.max_num_seqs,
            args.max_num_batched_tokens,
        )
        return

    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = DEFAULT_GPU_MEMORY_UTILIZATION
    if args.max_num_seqs is None:
        args.max_num_seqs = DEFAULT_MAX_NUM_SEQS


def validate_model_input(model: str) -> None:
    """Reject inputs that this launch path cannot interpret as HF models."""

    if Path(model).suffix.lower() == ".gguf":
        raise ValueError(GGUF_UNSUPPORTED_MESSAGE)


def parse_tool_calls(text: str):
    """
    Parse tool calls from LLM output text.

    Supports multiple formats used by different models:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>  (Qwen2.5, Hermes)
    2. {"tool": "...", "arguments": {...}}  (generic JSON)
    3. {"name": "...", "arguments": {...}}  (direct JSON)

    Returns list of tool call dicts or None if no tool calls found.
    """
    tool_calls = []

    # Format 1: <tool_call> tags (Qwen2.5, Hermes format)
    tag_matches = re.findall(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>', text)
    for match in tag_matches:
        try:
            parsed = json.loads(match)
            name = parsed.get("name") or parsed.get("function", {}).get("name")
            args = parsed.get("arguments") or parsed.get("parameters") or parsed.get("function", {}).get("arguments", {})
            if name:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Format 2: Look for JSON objects with tool/name + arguments fields
    # Find all JSON objects in the text
    for pattern in [
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:',
        r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:',
    ]:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            # Find the complete JSON object by counting braces
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                parsed = json.loads(text[start:end])
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name:
                    tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                continue

    return tool_calls if tool_calls else None


def build_tool_system_prompt(tools: list) -> str:
    """
    Build a system prompt section describing available tools.
    Used as fallback when the model's chat template doesn't support tools.
    """
    lines = [
        "You have access to the following tools. To use a tool, respond with a JSON object:",
        '{"name": "tool_name", "arguments": {"param": "value"}}',
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_strs = []
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            param_strs.append(f"    - {pname}: {pinfo.get('type', 'string')}{req} - {pinfo.get('description', '')}")

        lines.append(f"- {name}: {desc}")
        if param_strs:
            lines.extend(param_strs)
    lines.append("")
    lines.append("If you don't need a tool, respond with plain text.")

    return "\n".join(lines)


def format_tool_calls_response(tool_calls: list) -> list:
    """Format parsed tool calls into OpenAI tool_calls response format."""
    formatted = []
    for i, tc in enumerate(tool_calls):
        formatted.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
            },
        })
    return formatted


def create_app(llm: LLM, model_name: str, task: str = "generate"):
    """Create FastAPI app with OpenAI-compatible endpoints."""
    from vllm import SamplingParams
    app = FastAPI()
    engine = llm.llm_engine

    # One dispatcher owns engine.step() and routes every request's outputs.
    dispatcher = EngineDispatcher(engine, logger)
    dispatcher_task: asyncio.Task | None = None

    @app.on_event("startup")
    async def startup():
        nonlocal dispatcher_task
        dispatcher_task = asyncio.create_task(dispatcher.run())

    @app.on_event("shutdown")
    async def shutdown():
        dispatcher.stop()
        if dispatcher_task is not None:
            await dispatcher_task

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "vllm",
            }]
        })

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if task != "embed":
            return JSONResponse(
                {"error": "This server is running in 'generate' mode. Start with --task embed for embeddings."},
                status_code=400,
            )

        body = await request.json()
        input_texts = body.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        if not isinstance(input_texts, list) or not input_texts:
            return JSONResponse({"error": "input must be a non-empty string or list"}, status_code=400)

        request_id = f"embd-{uuid.uuid4().hex[:12]}"

        try:
            outputs = llm.embed(input_texts, use_tqdm=False)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        data = []
        total_tokens = 0
        for i, output in enumerate(outputs):
            embedding = output.outputs.embedding
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            num_tokens = len(output.prompt_token_ids)
            total_tokens += num_tokens
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            })

        return JSONResponse({
            "id": request_id,
            "object": "list",
            "model": model_name,
            "data": data,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if task != "generate":
            return JSONResponse(
                {"error": "This server is running in 'embed' mode."}, status_code=400
            )
        body = await request.json()

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        stream = body.get("stream", False)
        stop = body.get("stop")
        top_k = body.get("top_k", -1)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        tools = body.get("tools")
        response_format = body.get("response_format")

        # Build prompt from messages using tokenizer's chat template
        tokenizer = llm.get_tokenizer()
        if tools:
            # Strategy 1: Try native tools via chat template
            native_ok = False
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True
                )
                # Verify tools actually appeared in the prompt
                # (some templates silently accept but ignore the tools kwarg)
                tool_name = tools[0].get("function", tools[0]).get("name", "")
                if tool_name and tool_name in prompt:
                    native_ok = True
            except Exception:
                pass

            # Strategy 2: Inject tool descriptions into the system prompt
            if not native_ok:
                tool_prompt = build_tool_system_prompt(tools)
                augmented_messages = list(messages)
                if augmented_messages and augmented_messages[0].get("role") == "system":
                    augmented_messages[0] = {
                        **augmented_messages[0],
                        "content": augmented_messages[0]["content"] + "\n\n" + tool_prompt,
                    }
                else:
                    augmented_messages.insert(0, {"role": "system", "content": tool_prompt})

                try:
                    prompt = tokenizer.apply_chat_template(
                        augmented_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Final fallback: simple concatenation
                    parts = []
                    for m in augmented_messages:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content if p.get("type") == "text"
                            )
                        parts.append(f"{role}: {content}")
                    parts.append("assistant:")
                    prompt = "\n".join(parts)
        else:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content if p.get("type") == "text"
                        )
                    parts.append(f"{role}: {content}")
                parts.append("assistant:")
                prompt = "\n".join(parts)

        # Note: structured_outputs (response_format) is not supported on Windows
        # V1 engine with InprocClient — causes "sample_tokens()" state error.
        # The caller should handle JSON parsing/repair on its own.
        if response_format:
            logger.debug("response_format requested but not supported on Windows V1 engine; ignoring")

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if not stream:
            try:
                output_queue = dispatcher.add_request(request_id, prompt, params)
                while True:
                    output = await output_queue.get()
                    if isinstance(output, EngineDispatchFailure):
                        raise output.error
                    if output.finished:
                        break
            except asyncio.CancelledError:
                dispatcher.abort(request_id)
                raise
            except Exception as exc:
                dispatcher.abort(request_id)
                logger.exception("Request %s failed", request_id)
                return JSONResponse({"error": str(exc)}, status_code=500)
            finally:
                dispatcher.finish(request_id)

            text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            prompt_tokens = len(output.prompt_token_ids)

            # Check for tool calls in the response
            message = {"role": "assistant", "content": text}
            finish_reason = output.outputs[0].finish_reason or "stop"

            if tools:
                parsed_tools = parse_tool_calls(text)
                if parsed_tools:
                    message["tool_calls"] = format_tool_calls_response(parsed_tools)
                    message["content"] = None
                    finish_reason = "tool_calls"

            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "completion_tokens": tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + tokens,
                },
            })

        try:
            output_queue = dispatcher.add_request(request_id, prompt, params)
        except Exception as exc:
            logger.exception("Could not add request %s", request_id)
            return JSONResponse({"error": str(exc)}, status_code=500)

        async def stream_response():
            rid = request_id
            prev_len = 0
            completed = False
            try:
                while True:
                    output = await output_queue.get()
                    if isinstance(output, EngineDispatchFailure):
                        raise output.error
                    text = output.outputs[0].text
                    new_text = text[prev_len:]
                    prev_len = len(text)

                    if new_text:
                        chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": new_text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if output.finished:
                        completed = True
                        finish_chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": output.outputs[0].finish_reason or "stop",
                            }],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Streaming request %s failed", rid)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if completed:
                    dispatcher.finish(rid)
                else:
                    dispatcher.abort(rid)

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ---------------------------------------------------------------------------
#  Interactive model discovery and selection
# ---------------------------------------------------------------------------

CONFIG_FILE = ".vllm_config.json"


def _load_config() -> dict:
    """Load saved config (last model selection, etc.)."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict):
    """Save config to disk."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def _dir_size_gb(path: Path) -> float:
    """Estimate directory size in GB."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total / (1024 ** 3)


def find_models(search_paths: list[str | Path] | None = None) -> list[dict]:
    """
    Scan directories for Hugging Face-format model directories.

    Returns list of dicts: {path, name, model_type, size_gb, kind}
    """
    models = []
    seen_paths: set[str] = set()

    # Build search paths
    dirs_to_scan: list[Path] = []

    if search_paths:
        for p in search_paths:
            d = Path(p)
            if d.is_dir():
                dirs_to_scan.append(d)

    # HuggingFace cache
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if hf_cache.is_dir():
        dirs_to_scan.append(hf_cache)

    # Scan common locations on Windows drives
    for drive in ["C:", "D:", "E:", "F:", "G:"]:
        for subdir in ["models", "LLM", "llm", "huggingface"]:
            candidate = Path(f"{drive}\\{subdir}")
            if candidate.is_dir():
                dirs_to_scan.append(candidate)

    for scan_dir in dirs_to_scan:
        try:
            _scan_directory(scan_dir, models, seen_paths, max_depth=3)
        except OSError:
            continue

    # Sort by name
    models.sort(key=lambda m: m["name"].lower())
    return models


def _scan_directory(directory: Path, models: list, seen: set, max_depth: int, depth: int = 0):
    """Recursively scan a directory for models."""
    if depth > max_depth:
        return

    resolved = str(directory.resolve())
    if resolved in seen:
        return
    seen.add(resolved)

    try:
        entries = list(directory.iterdir())
    except OSError:
        return

    # Check if this directory itself is a HuggingFace model
    config_json = directory / "config.json"
    if config_json.is_file():
        try:
            with open(config_json, "r") as f:
                config = json.load(f)
            model_type = config.get("model_type", "unknown")
            models.append({
                "path": str(directory),
                "name": directory.name,
                "model_type": model_type,
                "size_gb": round(_dir_size_gb(directory), 1),
                "kind": "hf",
            })
            return  # Don't recurse into model directories
        except (json.JSONDecodeError, OSError):
            pass

    # Check for HF cache snapshot directories (models--org--name/snapshots/hash/)
    if directory.name.startswith("models--"):
        snapshots_dir = directory / "snapshots"
        if snapshots_dir.is_dir():
            try:
                snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for snap in snapshot_dirs[:1]:  # Latest snapshot only
                    if (snap / "config.json").is_file():
                        _scan_directory(snap, models, seen, max_depth=0, depth=0)
                        # Fix the name from the snapshot
                        if models and models[-1]["path"] == str(snap):
                            hf_name = directory.name.replace("models--", "").replace("--", "/")
                            models[-1]["name"] = hf_name
            except OSError:
                pass
            return

    for entry in entries:
        if entry.is_dir() and not entry.name.startswith("."):
            _scan_directory(entry, models, seen, max_depth, depth + 1)


def interactive_model_select(search_paths: list[str | Path] | None = None) -> str:
    """
    Display an interactive model selector and return the chosen model path.
    """
    config = _load_config()
    last_model = config.get("last_model")

    print("\n" + "=" * 60)
    print("  vLLM Windows - Model Selection")
    print("=" * 60)
    print("\nScanning for models...")

    models = find_models(search_paths)

    if not models:
        print("\nNo models found automatically.")
        print("Enter a Hugging Face model directory or repository ID:")
        path = input("> ").strip().strip('"')
        if not path:
            print("No model specified. Exiting.")
            sys.exit(1)
        _save_config({"last_model": path})
        return path

    # Display table
    print(f"\nFound {len(models)} model(s):\n")
    # Find index of last model for default marker
    default_idx = None
    for i, m in enumerate(models):
        if m["path"] == last_model:
            default_idx = i
            break

    print(f"  {'#':>3}  {'Model':<40}  {'Type':<10}  {'Size':>7}  Path")
    print(f"  {'---':>3}  {'----':<40}  {'----':<10}  {'----':>7}  ----")
    for i, m in enumerate(models):
        marker = " *" if i == default_idx else ""
        size_str = f"{m['size_gb']:.1f} GB" if m['size_gb'] > 0 else "?"
        name = m["name"][:40]
        print(f"  [{i + 1:>2}] {name:<40}  {m['model_type']:<10}  {size_str:>7}  {m['path']}{marker}")

    print(f"\n  [ 0] Enter path manually")
    if default_idx is not None:
        print(f"\n  * = last used (press Enter to reuse)")

    # Get selection
    while True:
        prompt = "\nSelect model number"
        if default_idx is not None:
            prompt += f" [{default_idx + 1}]"
        prompt += ": "

        choice = input(prompt).strip()

        if choice == "" and default_idx is not None:
            selected = models[default_idx]["path"]
            break

        try:
            num = int(choice)
        except ValueError:
            print("  Invalid input. Enter a number.")
            continue

        if num == 0:
            print("Enter model path:")
            selected = input("> ").strip().strip('"')
            if not selected:
                print("  No path entered.")
                continue
            break
        elif 1 <= num <= len(models):
            selected = models[num - 1]["path"]
            break
        else:
            print(f"  Invalid choice. Enter 0-{len(models)}.")

    _save_config({"last_model": selected})
    print(f"\nSelected: {selected}\n")
    return selected


def main():
    parser = argparse.ArgumentParser(description="vLLM Windows Server")
    parser.add_argument("--model", default=None, help="Model path or HF ID (interactive selector if omitted)")
    parser.add_argument("--models-dir", default=None,
                        help="Additional directory to scan for models")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max context length (default: 8192). This caps the KV-cache "
                             "size: vLLM pre-allocates a paged KV cache sized to "
                             "max_model_len, so a model advertising 128K context would "
                             "otherwise reserve many GB of VRAM up front. Raise for "
                             "long-context use, lower to save VRAM. Pass 0 to use the "
                             "model's full advertised length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Maximum fraction of total GPU memory vLLM may use for weights "
                             "and KV cache (default: 0.6; 0.89 with --turing-compat). Raise "
                             "it when weights fit but the KV-cache budget is negative; lower "
                             "it only when reserving VRAM for other applications.")
    parser.add_argument("--max-num-seqs", "--max-num-seq", type=int, default=None,
                        help="Max concurrent sequences (default: 64; 1 with "
                             "--turing-compat). Lower it to reduce startup and KV-cache "
                             "requirements.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs (reduces throughput; useful for compatibility/debugging)")
    parser.add_argument("--attention-backend", default=None,
                        help="Force an attention backend (for RTX 20xx/Turing, use TRITON_ATTN; default: auto)")
    parser.add_argument("--linear-backend", default=None,
                        help="Linear kernel backend (for example marlin; default: auto)")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="KV-cache dtype (auto, float16, bfloat16, fp8, or a supported TurboQuant dtype)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="KV-cache block size in tokens (16 or 32 are conservative compatibility choices)")
    parser.add_argument("--turing-compat", action="store_true",
                        help="Tested RTX 20xx/SM 7.5 profile: TRITON_ATTN, float16 KV, "
                             "block 32, eager mode, 0.89 GPU memory, 1 sequence, and a "
                             "2,048-token batch cap")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU index to use; with launch.bat this follows nvidia-smi (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="Enable automatic prefix caching (reuses KV cache for shared prefixes)")
    parser.add_argument("--num-scheduler-steps", type=int, default=1,
                        help="Legacy compatibility option; only 1 is supported")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Max tokens per scheduler iteration (default: auto; 2048 with "
                             "--turing-compat; higher values improve prefill throughput but "
                             "need more memory)")
    parser.add_argument("--task", default="generate", choices=["generate", "embed"],
                        help="Task type: 'generate' for text generation, 'embed' for embeddings")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom model code (needed for some embedding models like nomic-bert)")
    parser.add_argument("--cpu-offload-gb", type=float, default=0,
                        help="GiB of model weights to offload to CPU (useful for Windows WDDM memory limits)")
    parser.add_argument(
        "--kv-offload",
        choices=("disabled", "cpu-lru", "cpu-arc", "fs-lru", "fs-arc"),
        default="disabled",
        help=(
            "Experimental prompt-KV offload mode (default: disabled). "
            "CPU modes use pinned RAM; fs modes add a persistent filesystem tier."
        ),
    )
    parser.add_argument(
        "--kv-offload-cpu-gb",
        type=float,
        default=4.0,
        help=(
            "GiB for the KV offload RAM tier when --kv-offload is enabled "
            "(default: 4.0; separate from --cpu-offload-gb model-weight offload)"
        ),
    )
    parser.add_argument(
        "--kv-offload-fs-root",
        default=None,
        help=(
            "Cache directory required by fs-lru/fs-arc. Files persist across "
            "runs and have no automatic size quota."
        ),
    )
    parser.add_argument(
        "--kv-offload-read-threads",
        type=int,
        default=4,
        help="Read-priority filesystem cache threads (default: 4)",
    )
    parser.add_argument(
        "--kv-offload-write-threads",
        type=int,
        default=4,
        help="Write-priority filesystem cache threads (default: 4)",
    )
    args = parser.parse_args()

    if args.num_scheduler_steps != 1:
        parser.error(
            "--num-scheduler-steps values other than 1 are not supported by "
            "the current vLLM engine; remove this option"
        )

    # These defaults include the values confirmed by the issue #14 reporter
    # on an RTX 2080 Ti with 11 GiB. Explicit flags still win.
    apply_runtime_defaults(args)

    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be at least 1")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        parser.error("--max-num-batched-tokens must be at least 1")
    if args.kv_offload != "disabled" and args.kv_offload_cpu_gb <= 0:
        parser.error("--kv-offload-cpu-gb must be greater than zero")
    if args.kv_offload_read_threads < 1:
        parser.error("--kv-offload-read-threads must be at least 1")
    if args.kv_offload_write_threads < 1:
        parser.error("--kv-offload-write-threads must be at least 1")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    if args.kv_offload.startswith("fs-") and not args.kv_offload_fs_root:
        parser.error("--kv-offload-fs-root is required for fs-lru/fs-arc")
    if args.kv_offload_fs_root and not args.kv_offload.startswith("fs-"):
        parser.error("--kv-offload-fs-root is only valid with fs-lru/fs-arc")

    # Interactive model selection if --model not provided
    if args.model is None:
        search_paths = [args.models_dir] if args.models_dir else None
        args.model = interactive_model_select(search_paths)

    try:
        validate_model_input(args.model)
    except ValueError as exc:
        parser.error(
            str(exc)
        )

    # Pin to specific GPU if requested
    if args.gpu_id is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logger.info(f"Pinned to GPU {args.gpu_id}")

    logger.info(f"Loading model: {args.model}")
    ctx_desc = "model default (full)" if not args.max_model_len else str(args.max_model_len)
    logger.info(
        f"Memory settings: gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_model_len={ctx_desc}, max_num_seqs={args.max_num_seqs}. "
        f"If weights load but vLLM reports no available KV-cache memory, raise "
        f"--gpu-memory-utilization when VRAM is free, or lower --max-model-len, "
        f"--max-num-seqs, and --max-num-batched-tokens."
    )
    start = time.time()

    llm_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    # Only force eager mode if explicitly requested (it disables CUDA graphs)
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.attention_backend:
        # Import lazily so the default path remains compatible with older
        # packaged vLLM versions that predate AttentionConfig.
        from vllm.config import AttentionConfig
        llm_kwargs["attention_config"] = AttentionConfig(backend=args.attention_backend)
    if args.kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.block_size:
        llm_kwargs["block_size"] = args.block_size
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.enable_prefix_caching or args.kv_offload != "disabled":
        llm_kwargs["enable_prefix_caching"] = True
    if args.linear_backend:
        llm_kwargs["linear_backend"] = args.linear_backend.lower().replace("-", "_")
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.kv_offload != "disabled":
        from vllm.config import KVTransferConfig

        policy = args.kv_offload.rsplit("-", maxsplit=1)[1]
        extra_config = {
            "cpu_bytes_to_use": int(args.kv_offload_cpu_gb * (1 << 30)),
            "eviction_policy": policy,
            "offload_prompt_only": True,
        }
        if args.kv_offload.startswith("fs-"):
            fs_root = Path(args.kv_offload_fs_root).expanduser().resolve()
            if fs_root.exists() and not fs_root.is_dir():
                parser.error(f"--kv-offload-fs-root is not a directory: {fs_root}")
            extra_config.update(
                {
                    "spec_name": "TieringOffloadingSpec",
                    "secondary_tiers": [
                        {
                            "type": "fs",
                            "root_dir": str(fs_root),
                            "n_read_threads": args.kv_offload_read_threads,
                            "n_write_threads": args.kv_offload_write_threads,
                        }
                    ],
                }
            )
            if os.environ.get("PYTHONHASHSEED") != "0":
                logger.warning(
                    "Filesystem KV cache is enabled without PYTHONHASHSEED=0 "
                    "at process start. In-process reuse still works, but cache "
                    "entries may not be reusable after a restart. launch.bat "
                    "sets the required value automatically."
                )
            logger.warning(
                "Filesystem KV cache root %s has no automatic size quota; "
                "monitor and clean it explicitly.",
                fs_root,
            )
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        )
        logger.info(
            "Experimental KV offload enabled: mode=%s, RAM tier=%.2f GiB; "
            "prefix caching enabled",
            args.kv_offload,
            args.kv_offload_cpu_gb,
        )

    from vllm import LLM

    llm = LLM(**llm_kwargs)
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    model_name = os.path.basename(args.model)
    app = create_app(llm, model_name, task=args.task)

    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                loop="asyncio")


if __name__ == "__main__":
    main()
