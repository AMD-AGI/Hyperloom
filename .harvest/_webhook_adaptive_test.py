"""Minimal Adaptive Card test for the Power Automate Workflows webhook."""
import json
import os
import sys

import requests
import urllib3
urllib3.disable_warnings()

url = os.environ.get(
    "WEBHOOK_URL",
    "https://default3dd8961fe4884e608e11a82d994e18.3d.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/e3c96d479c2a4338ab09432755abc1cf/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=EHgChEMxN6eAeeBxmpG1pt6jAPiBGtyzYe2hMOhzJLI",
)

payload = {
    "type": "message",
    "attachments": [
        {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "Hyperloom CI webhook test",
                        "weight": "Bolder",
                        "size": "Medium",
                    },
                    {
                        "type": "TextBlock",
                        "text": "If you see this card in Teams, Adaptive Card payload works.",
                        "wrap": True,
                    },
                ],
            },
        }
    ],
}

print("Payload:")
print(json.dumps(payload, indent=2))
print()

resp = requests.post(url, json=payload, verify=False, timeout=10)
print(f"status={resp.status_code}")
print(f"body={resp.text[:500]!r}")
sys.exit(0 if resp.status_code < 300 else 1)
