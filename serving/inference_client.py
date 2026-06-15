import json
import os
import sys
from pathlib import Path

import requests


def main(payload_path: str) -> None:
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        raise RuntimeError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set.")

    endpoint_name = os.environ.get("DATABRICKS_ENDPOINT_NAME", "credit-risk-endpoint-prod")
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(30, 180),
    )

    print(response.status_code)
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else "serving/test_payload.json"
    main(payload)
