Hyperloom documentation
========================

Hyperloom is an agentic system that autonomously optimizes LLM inference on AMD
GPU platforms. It treats optimization as a search problem: given a workload, it
explores candidate optimizations one change at a time, always measuring against
the real workload and using prior results plus KB priors to choose the next
move. This site is generated with Sphinx; the **API Reference** is built
automatically from the in-code Google-style docstrings via ``autodoc`` and
``napoleon``.

.. toctree::
   :maxdepth: 1
   :caption: Hyperloom

   overview
   release_notes
   compatibility
   components/index

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: How-to Guides

   how_to_optimize
   HOW_THE_OPTIMIZATION_LOOP_WORKS
   OPERATIONS
   OPERATOR_SCRIPTS
   CONFIGURATION_REFERENCE
   ENV_AND_AUTH
   KB_GUIDE
   INTEGRATION_SESSION_BREAKDOWN
   TROUBLESHOOTING
   UPGRADING

.. toctree::
   :maxdepth: 1
   :caption: Case Studies

   CASE_STUDY_GLM5
   CASE_STUDY_DEEPSEEK_R1

.. toctree::
   :maxdepth: 1
   :caption: About

   about


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
