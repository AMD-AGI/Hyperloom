# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""First-pass tuning predictor: consult an external model at FRAMEWORK entry.

The pump builds a JSON document out of ``SharedState`` plus the trace-analysis
artifacts, POSTs it to a service that owns the prompt rendering and the model,
and turns the answer back into ordinary ``explore`` variants and free-form
specialist mandates. Nothing here evaluates a proposal: the existing explore /
``integrate_patch`` paths keep their own KEEP thresholds and accuracy gate, so
a predicted variant is measured on the same ruler as ``default_grid`` and
``llm_direct``.

The model deliberately runs elsewhere. IR-1 wants every visible GPU idle before
a serving launch, so a co-resident predictor would degrade the benchmark it is
meant to improve.

See ``docs/reference/primatune-predictor.md`` for the wire contract.
"""
