"""
EventLogger - Persistent training events to events.jsonl

Writes training metrics, logs, samples, and checkpoints to JSONL file
for historical replay and analysis.
"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime


class EventLogger:
    """
    Logs training events to events.jsonl file.
    Flushes after each write for immediate visibility.
    """

    def __init__(self, events_file: str):
        """
        Args:
            events_file: Path to events.jsonl (will be created/appended)
        """
        self.events_file = events_file
        os.makedirs(os.path.dirname(events_file), exist_ok=True)
        self.file = open(events_file, 'a', encoding='utf-8')

    def log_metric(self, epoch: int, metrics: Dict[str, Any]):
        """Log training/validation metrics for an epoch."""
        event = {
            "type": "metric",
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self._write(event)

    def log_status(self, status: str, message: str = ""):
        """Log status change (RUNNING/FINISHED/FAILED)."""
        event = {
            "type": "status",
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self._write(event)

    def log_checkpoint(self, kind: str, path: str, epoch: int, metrics: Dict[str, Any]):
        """Log checkpoint save event."""
        event = {
            "type": "checkpoint",
            "kind": kind,
            "path": path.replace("\\", "/"),
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self._write(event)

    def log_sample_pred(self, epoch: int, items: List[Dict[str, Any]]):
        """Log sample predictions WITH images (base64)."""
        event = {
            "type": "sample_pred",
            "epoch": epoch,
            "count": len(items),
            "items": [
                {
                    "id": item["id"],
                    "target": item["target"],
                    "pred": item["pred"],
                    "ok": item.get("ok", False),
                    "image_b64": item.get("image_b64", "")  # Include images!
                }
                for item in items
            ],
            "timestamp": datetime.now().isoformat()
        }
        self._write(event)

    def log_text(self, line: str):
        """Log text message."""
        event = {
            "type": "log",
            "ts": datetime.now().isoformat(),
            "line": line
        }
        self._write(event)

    def log_dataset_stats(self, train_samples: int, val_samples: int, stats: Dict[str, Any]):
        """Log dataset statistics."""
        event = {
            "type": "dataset_stats",
            "train_samples": train_samples,
            "val_samples": val_samples,
            "timestamp": datetime.now().isoformat(),
            **stats
        }
        self._write(event)

    def _write(self, event: Dict[str, Any]):
        """Write event to file and flush immediately."""
        self.file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        """Close the file handle."""
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_events(events_file: str) -> List[Dict[str, Any]]:
    """
    Read all events from events.jsonl file.

    Returns:
        List of event dictionaries, chronological order.
    """
    if not os.path.exists(events_file):
        return []

    events = []
    with open(events_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue

    return events


def parse_events_by_type(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group events by type.

    Returns:
        {
            "metric": [...],
            "log": [...],
            "checkpoint": [...],
            "sample_pred": [...],
            "status": [...]
        }
    """
    grouped = {
        "metric": [],
        "log": [],
        "checkpoint": [],
        "sample_pred": [],
        "status": [],
        "dataset_stats": []
    }

    for event in events:
        event_type = event.get("type", "log")
        if event_type in grouped:
            grouped[event_type].append(event)
        else:
            grouped["log"].append(event)

    return grouped
