"""Contract tests for the Forge KB implementation identity."""

from __future__ import annotations

import pytest

from kernelforge.knowledge.implementation_identity import (
    canonical_editable_source_paths,
    canonical_framework_version,
    canonical_owner_framework,
    implementation_signature,
    normalize_operator_name,
)


def test_operator_name_is_logical_and_backend_prefix_independent():
    assert normalize_operator_name("backend::Fused.MoE-Kernel") == "fused_moe"


def test_the_same_kernel_spelled_either_way_is_one_operator():
    # The reported split: a header declares KdaPackedDecodeKernel, the module binding it
    # defines kda_packed_decode_kernel, and each spelling addressed a page the other never
    # wrote to -- so the second campaign re-derived a port the first had already validated.
    assert normalize_operator_name("KdaPackedDecodeKernel") == "kda_packed_decode"
    assert normalize_operator_name("kda_packed_decode_kernel") == "kda_packed_decode"
    assert normalize_operator_name("kda_packed_decode") == "kda_packed_decode"


def test_namespaced_camel_case_matches_its_snake_case_spelling():
    # The task contract asks for this spelling, so upstream pull-request search has term
    # boundaries to split on. It may not cost the operator its page to supply one.
    assert normalize_operator_name("aiter::fusedAddRmsNorm") == normalize_operator_name("fused_add_rms_norm")
    assert normalize_operator_name("SiluAndMul") == normalize_operator_name("silu_and_mul")


def test_an_acronym_run_is_one_word():
    # Splitting on every case change would cut MoE into mo_e and QKV into q_k_v, inventing
    # a difference between spellings of one kernel instead of removing one.
    assert normalize_operator_name("FusedMoE") == "fused_moe"
    assert normalize_operator_name("KVCache") == "kv_cache"
    assert normalize_operator_name("paged_attention_ll4mi_QKV_mfma16_kernel") == "paged_attention_ll4mi_qkv_mfma16"
    assert normalize_operator_name("HGEMV_WFPerRow") == "hgemv_wf_per_row"


def test_a_capital_after_a_whole_word_is_its_own_word():
    # The counterpart to the acronym rule. Absorbing every trailing capital to keep MoE
    # whole also welded the dimension letters these kernels end in onto the word before
    # them, which disagreed with the snake spelling the same source declares: measured
    # across the store's symbols and this tree, 82 of 769 kernel names.
    assert normalize_operator_name("ChunkFwdKernelO") == normalize_operator_name("chunk_fwd_kernel_o")
    assert normalize_operator_name("IndexerKQuantAndCache") == normalize_operator_name("indexer_k_quant_and_cache")
    assert normalize_operator_name("AllreduceMhcPostLargeMKernel") == "allreduce_mhc_post_large_m"


def test_a_digit_run_keeps_whatever_boundary_the_spelling_gave_it():
    """Known gap, pinned so a change in it cannot pass unnoticed.

    Whether ``2stage`` is one word or two is not recoverable from the spelling:
    ``mxfp4_moe_2stage`` says one, ``Mxfp4Moe2Stage`` says two, and a rule that
    picked either would re-address the pages the other spelling wrote. The store
    holds 8 such names, all spelled as words already, so the ambiguity costs a
    page only if some later campaign supplies the camel spelling of one.
    """
    assert normalize_operator_name("mxfp4_moe_2stage") == "mxfp4_moe_2stage"
    assert normalize_operator_name("Mxfp4Moe2stage") == "mxfp4_moe2stage"
    assert normalize_operator_name("gemm_a8w8_blockscale") == "gemm_a8w8_blockscale"


def test_names_already_written_as_words_are_left_alone():
    # Every page in the store is addressed by one of these; re-spelling them would strand
    # the histories they hold.
    for name in (
        "unified_attention_with_output",
        "gemm_a8w8_blockscale_bpreshuffle",
        "mxfp4_moe_2stage_t16",
        "custom_all_reduce_tp8",
        "rocm_unquantized_gemm",
    ):
        assert normalize_operator_name(name) == name


def test_distinct_operators_sharing_a_prefix_stay_distinct():
    assert normalize_operator_name("MoeFlydslStage1") != normalize_operator_name("MoeFlydslStage2")
    assert normalize_operator_name("rmsNorm") != normalize_operator_name("addRmsNorm")


def test_the_build_a_release_was_compiled_as_is_not_part_of_the_release():
    # The store holds unified_attention_with_output under 0.24.0 and 0.24.0+rocm723
    # with one implementation_signature between them: one kernel, two pages, and the
    # faster port (11.58x) on the page the other spelling never reads.
    assert canonical_framework_version("0.24.0+rocm723") == "0.24.0"
    assert canonical_framework_version("0.1.dev19253+g5f76ae224.d20260727.rocm723") == "0.1"


def test_a_release_reads_the_same_however_it_was_written_down():
    # importlib.metadata reports the distribution version; a campaign reads the tag
    # or the image it arrived in. Both are naming the source a port was written
    # against, so both have to address one page.
    assert canonical_framework_version("v0.24.0") == canonical_framework_version("0.24.0")
    assert canonical_framework_version("v0.5.15.post1-rocm720-mi35x-20260724") == "0.5.15.post1"
    assert canonical_framework_version("0.5.15.post1.dev20260724+g3d91a569ce") == "0.5.15.post1"


def test_every_word_for_not_knowing_the_version_is_the_same_word():
    # Three code paths answer "which version?" in three different words, and the
    # store holds rmsnorm under two of them -- a 1.49x port and a 1.15x port of one
    # kernel, neither page reachable from the other.
    words = {"", "none", "unknown", "unspecified", "unknown_version"}
    assert {canonical_framework_version(word) for word in words} == {"unknown"}


def test_distinct_releases_stay_distinct():
    assert canonical_framework_version("0.11.3") != canonical_framework_version("0.11.4")
    assert canonical_framework_version("0.5.15") != canonical_framework_version("0.5.15.post1")
    assert canonical_framework_version("1.0.0rc1") != canonical_framework_version("1.0.0")


def test_a_version_naming_no_release_is_left_exactly_as_written():
    # Read side and write side canonicalize independently, so a string this cannot
    # read has to survive unchanged rather than become some other page's name.
    for raw in ("not-a-version", "0.24.0+rocm723", "unspecified"):
        once = canonical_framework_version(raw)
        assert canonical_framework_version(once) == once
    assert canonical_framework_version("not-a-version") == "not-a-version"


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


def test_signature_does_not_publish_partial_symbols_after_extraction_failure(tmp_path, monkeypatch):
    import kernelforge.mcp_server.tools.pmc as pmc

    def fail_extraction(_source):
        raise RuntimeError("kernel parser failed")

    monkeypatch.setattr(pmc, "derive_kernel_names", fail_extraction)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("@triton.jit\ndef target_kernel(x):\n    return x\n")

    with pytest.raises(RuntimeError, match="kernel parser failed"):
        implementation_signature(
            workspace=str(tmp_path),
            kernel_path=str(kernel),
            source_files=[],
            framework="unknown",
        )
