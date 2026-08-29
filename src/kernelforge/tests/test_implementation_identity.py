"""Contract tests for the Forge KB implementation identity."""

from __future__ import annotations

from kernelforge.knowledge.implementation_identity import (
    canonical_editable_source_paths,
    canonical_owner_framework,
    implementation_signature,
    normalize_operator_name,
)


def test_operator_name_is_logical_and_backend_prefix_independent():
    assert normalize_operator_name("backend::Fused.MoE-Kernel") == "fused_moe"


def test_operator_name_strips_balanced_nested_template_arguments():
    raw = "backend::paged_attention<half, layout<16, 8>>_kernel"
    assert normalize_operator_name(raw) == "paged_attention"
    assert normalize_operator_name(raw) == normalize_operator_name("paged_attention")


def test_package_relative_paths_ignore_workspace_layout(tmp_path):
    producer = tmp_path / "producer" / "vllm" / "ops" / "kernel.py"
    consumer = tmp_path / "consumer" / "src" / "vllm" / "ops" / "kernel.py"
    producer.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    source = "import triton\n@triton.jit\ndef fused_kernel(x):\n    return x\n"
    producer.write_text(source)
    consumer.write_text(source)

    producer_signature, producer_identity = implementation_signature(
        workspace=str(tmp_path / "producer"),
        kernel_path=str(producer),
        source_files=[],
        framework="vllm",
    )
    consumer_signature, consumer_identity = implementation_signature(
        workspace=str(tmp_path / "consumer"),
        kernel_path=str(consumer),
        source_files=[],
        framework="vllm",
    )

    assert producer_signature == consumer_signature
    assert producer_identity == consumer_identity
    assert producer_identity == {
        "source_paths": ["vllm/ops/kernel.py"],
        "implementation_symbols": ["fused_kernel"],
    }


def test_signature_changes_with_path_or_concrete_symbol(tmp_path):
    first = tmp_path / "vllm" / "ops" / "kernel.py"
    second = tmp_path / "vllm" / "ops" / "other.py"
    first.parent.mkdir(parents=True)
    source = "import triton\n@triton.jit\ndef kernel_a():\n    pass\n"
    first.write_text(source)
    second.write_text(source)

    def signature(path):
        return implementation_signature(
            workspace=str(tmp_path),
            kernel_path=str(path),
            source_files=[],
            framework="vllm",
        )[0]

    base = signature(first)
    assert signature(second) != base
    first.write_text(source.replace("kernel_a", "kernel_b"))
    assert signature(first) != base


def test_standalone_paths_remain_workspace_relative(tmp_path):
    kernel = tmp_path / "src" / "kernel.py"
    kernel.parent.mkdir()
    kernel.write_text("def kernel():\n    pass\n")

    assert canonical_editable_source_paths(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="unknown",
    ) == ["kernel.py"]


def test_owner_alias_and_optional_src_layouts_converge(tmp_path):
    producer = tmp_path / "producer" / "src" / "aiter_meta" / "ops" / "kernel.py"
    consumer = tmp_path / "consumer" / "aiter" / "ops" / "kernel.py"
    producer.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    producer.write_text("def target():\n    pass\n")
    consumer.write_text(producer.read_text())

    left, left_identity = implementation_signature(
        workspace=str(tmp_path / "producer"),
        kernel_path=str(producer),
        source_files=[],
        framework="aiter_meta",
    )
    right, right_identity = implementation_signature(
        workspace=str(tmp_path / "consumer"),
        kernel_path=str(consumer),
        source_files=[],
        framework="aiter",
    )

    assert canonical_owner_framework("aiter_meta") == "aiter"
    assert left == right
    assert left_identity == right_identity
    assert left_identity["source_paths"] == ["aiter/ops/kernel.py"]


def test_explicit_owner_stabilizes_flattened_optional_src_layout(tmp_path):
    producer = tmp_path / "producer" / "src" / "ops" / "kernel.py"
    consumer = tmp_path / "consumer" / "ops" / "kernel.py"
    producer.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    producer.write_text("def target():\n    pass\n")
    consumer.write_text(producer.read_text())

    producer_paths = canonical_editable_source_paths(
        workspace=str(tmp_path / "producer"),
        kernel_path=str(producer),
        source_files=[],
        framework="vllm",
    )
    consumer_paths = canonical_editable_source_paths(
        workspace=str(tmp_path / "consumer"),
        kernel_path=str(consumer),
        source_files=[],
        framework="vllm",
    )

    assert producer_paths == consumer_paths == ["vllm/ops/kernel.py"]


def test_direct_signature_reflects_current_source_symbols(tmp_path):
    kernel = tmp_path / "vllm" / "ops" / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n@triton.jit\ndef target_kernel(x):\n    return x\n")
    before, before_identity = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="vllm",
    )
    kernel.write_text(kernel.read_text() + "\n@triton.jit\ndef optimization_helper(x):\n    return x\n")
    after, after_identity = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="vllm",
    )

    assert before != after
    assert before_identity["implementation_symbols"] == ["target_kernel"]
    assert after_identity["implementation_symbols"] == [
        "optimization_helper",
        "target_kernel",
    ]


def test_signature_covers_all_editable_paths_and_source_symbols(tmp_path):
    kernel = tmp_path / "vllm" / "ops" / "kernel.py"
    helper = tmp_path / "vllm" / "ops" / "helper.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n@triton.jit\ndef target_kernel(x):\n    return x\n")
    helper.write_text("import triton\n@triton.jit\ndef helper_kernel(x):\n    return x\n")

    _, identity = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[str(helper)],
        framework="vllm",
    )

    assert identity == {
        "source_paths": [
            "vllm/ops/helper.py",
            "vllm/ops/kernel.py",
        ],
        "implementation_symbols": [
            "helper_kernel",
            "target_kernel",
        ],
    }


def test_signature_uses_empty_symbols_when_source_has_no_kernel_entry(tmp_path):
    kernel = tmp_path / "wrapper.py"
    kernel.write_text("def wrapper():\n    pass\n")

    _, identity = implementation_signature(
        workspace=str(tmp_path),
        kernel_path=str(kernel),
        source_files=[],
        framework="unknown",
    )

    assert identity["implementation_symbols"] == []
