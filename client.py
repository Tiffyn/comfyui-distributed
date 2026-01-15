#!/usr/bin/env python3
"""ComfyUI distributed queue client."""

import json
import uuid
import urllib.request
import urllib.error
from pathlib import Path


class ComfyUIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"

    def queue_prompt(self, workflow: dict, client_id: str = None) -> dict:
        """Submit a workflow to the queue.

        Args:
            workflow: ComfyUI workflow in API format
            client_id: Optional client ID for tracking

        Returns:
            Response with prompt_id
        """
        if client_id is None:
            client_id = str(uuid.uuid4())

        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    def queue_workflow_file(self, workflow_path: str, overrides: dict = None) -> dict:
        """Load workflow from file and submit to queue.

        Args:
            workflow_path: Path to workflow JSON file (API format)
            overrides: Optional dict of node_id -> {field: value} to override

        Returns:
            Response with prompt_id
        """
        workflow = json.loads(Path(workflow_path).read_text())

        if overrides:
            for node_id, fields in overrides.items():
                if node_id in workflow:
                    workflow[node_id]["inputs"].update(fields)

        return self.queue_prompt(workflow)

    def get_queue(self) -> dict:
        """Get current queue status."""
        with urllib.request.urlopen(f"{self.base_url}/queue") as resp:
            return json.loads(resp.read().decode())

    def get_history(self, prompt_id: str = None) -> dict:
        """Get execution history."""
        url = f"{self.base_url}/history"
        if prompt_id:
            url += f"/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())

    def cancel(self, prompt_id: str) -> None:
        """Cancel a queued prompt."""
        payload = {"delete": [prompt_id]}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/queue",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI queue client")
    parser.add_argument("--host", default="127.0.0.1", help="Frontend host")
    parser.add_argument("--port", type=int, default=8188, help="Frontend port")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # queue command
    queue_parser = subparsers.add_parser("queue", help="Queue a workflow")
    queue_parser.add_argument("workflow", help="Path to workflow JSON")
    queue_parser.add_argument("--override", "-o", nargs=2, action="append",
                              metavar=("NODE.FIELD", "VALUE"),
                              help="Override node input (e.g., -o 3.seed 12345)")

    # status command
    subparsers.add_parser("status", help="Show queue status")

    # history command
    history_parser = subparsers.add_parser("history", help="Show history")
    history_parser.add_argument("--id", help="Specific prompt ID")

    # cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel prompt")
    cancel_parser.add_argument("prompt_id", help="Prompt ID to cancel")

    args = parser.parse_args()
    client = ComfyUIClient(args.host, args.port)

    if args.command == "queue":
        overrides = {}
        if args.override:
            for node_field, value in args.override:
                node_id, field = node_field.split(".", 1)
                if node_id not in overrides:
                    overrides[node_id] = {}
                # Try to parse as JSON, fall back to string
                try:
                    overrides[node_id][field] = json.loads(value)
                except json.JSONDecodeError:
                    overrides[node_id][field] = value

        result = client.queue_workflow_file(args.workflow, overrides or None)
        print(json.dumps(result, indent=2))

    elif args.command == "status":
        result = client.get_queue()
        print(json.dumps(result, indent=2))

    elif args.command == "history":
        result = client.get_history(args.id)
        print(json.dumps(result, indent=2))

    elif args.command == "cancel":
        client.cancel(args.prompt_id)
        print(f"Cancelled: {args.prompt_id}")


if __name__ == "__main__":
    main()
