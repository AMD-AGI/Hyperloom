"""Start-up hook for the DLRMv4 training adapter (ruppell.adapters.dlrm).

Put this directory on PYTHONPATH (with ruppell's python/ directory) and set
RUPPELL_DLRM=1: every Python process then installs the adapter's post-import
patches before the trainer's modules load. The trainer spawns its ranks (fresh
interpreters), so a start-up hook is the one place that reaches them all.
Without RUPPELL_DLRM=1 it does nothing. It shadows the distribution's own
sitecustomize (Ubuntu's apport hook), which the trainer does not use.
"""

import os

if os.environ.get("RUPPELL_DLRM") == "1":
    try:
        from ruppell.adapters import dlrm as _ruppell_dlrm

        _ruppell_dlrm.install()
    except Exception as _e:  # noqa: BLE001  (never break interpreter start-up)
        import sys

        print(f"[ruppell] DLRMv4 adapter not installed in pid {os.getpid()}: {type(_e).__name__}: {_e}",
              file=sys.stderr, flush=True)
