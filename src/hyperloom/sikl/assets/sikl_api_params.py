# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Print, as a JSON list, the keyword arguments SIKL calls the entry of the schema file in argv[1] with. Runs
with a SIKL checkout on PYTHONPATH."""

import json
import sys

from sikl.core.contract.registry import ContractRegistry

contract = ContractRegistry.default().for_schema_json(json.loads(open(sys.argv[1], encoding="utf-8").read()))
print(json.dumps([param.name for param in contract.api_params]))
