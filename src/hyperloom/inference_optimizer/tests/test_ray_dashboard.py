# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Ray Dashboard HTTP client."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

# Stub httpx, imported by ray_dashboard at module load.
_httpx_stub = types.ModuleType("httpx")
_httpx_stub.Timeout = lambda **kwargs: kwargs
_httpx_stub.Client = MagicMock()
_httpx_stub.Response = MagicMock
sys.modules.setdefault("httpx", _httpx_stub)

from hyperloom.inference_optimizer.multi_node._internal import ray_dashboard  # noqa: E402


def test_dashboard_client_sends_bearer_token_when_configured():
    with patch.object(ray_dashboard.httpx, "Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        client = ray_dashboard.RayDashboardClient("10.0.0.1", token="tok-abc")
        assert client is not None
        mock_client_cls.assert_called_once()
        headers = mock_client_cls.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer tok-abc"
        client.close()


def test_dashboard_client_omits_authorization_without_token():
    with patch.object(ray_dashboard.httpx, "Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        client = ray_dashboard.RayDashboardClient("10.0.0.1")
        headers = mock_client_cls.call_args.kwargs["headers"]
        assert "Authorization" not in headers
        client.close()


def test_submit_job_includes_runtime_env_payload():
    with patch.object(ray_dashboard.httpx, "Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"submission_id": "sub-1"}
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with ray_dashboard.RayDashboardClient("10.0.0.2") as client:
            sub = client.submit_job(
                "echo hi",
                runtime_env={"env_vars": {"REMOVED_BACKEND_SRC": "/weka/legacy-backend"}},
            )
        assert sub == "sub-1"
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["runtime_env"] == {"env_vars": {"REMOVED_BACKEND_SRC": "/weka/legacy-backend"}}
        assert "entrypoint" in payload
