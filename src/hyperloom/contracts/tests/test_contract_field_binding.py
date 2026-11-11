# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measurement and recipe contracts validate with the shared field checks, bound to their own error."""

from __future__ import annotations

import pytest

from hyperloom.contracts import _validate, recipe
from hyperloom.contracts._validate import MeasurementError
from hyperloom.contracts.recipe import RecipeError
from hyperloom_common.contract_fields import ContractFields


@pytest.mark.parametrize(("module", "error"), [(_validate, MeasurementError), (recipe, RecipeError)])
def test_contracts_bind_the_shared_checks_to_their_own_error(module, error) -> None:
    assert module._FIELDS == ContractFields(error)
    assert module._text.__func__ is ContractFields.text
    with pytest.raises(error):
        module._mapping("not an object", "a document")
