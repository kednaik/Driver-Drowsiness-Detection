"""Log analysis utilities for the Driver Drowsiness Alert System.

This module provides small helpers to read the CSV logs produced by
the analyzer, compute simple summaries (counts by day/hour/event),
plot a timeline, and preview saved screenshots.

Usage (quick):
    from tools.log_analysis import read_logs, print_summary
    events = read_logs("logs/drowsiness_log.csv")
    print_summary(events)

If run as a script it prints a short summary for the default log file.
"""
from __future__ import annotations

import csv
import datetime
import os
from collections import defaultdict
from typing import List, Dict, Any

try:
    import matplotlib.pyplot as plt
    import cv2
except Exception:
    plt = None
    cv2 = None


def read_logs(log_file: str = "logs/drowsiness_log.csv") -> List[Dict[str, Any]]:
    """Reads the CSV log file and returns a list of event dicts.

    Expected CSV columns: Timestamp, Event, EAR, MAR, Screenshot (optional)
    """
    events = []
    if not os.path.exists(log_file):
        return events

    with open(log_file, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return events

        # Map header indexes for flexibility
        hdr = {name: idx for idx, name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            ts_str = row[hdr.get("Timestamp", 0)]
            try:
                ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                # If timestamp parsing fails, skip row
                continue
            event = row[hdr.get("Event", 1)]
            ear = float(row[hdr.get("EAR", 2)]) if len(row) > hdr.get("EAR", 2) else None
            mar = float(row[hdr.get("MAR", 3)]) if len(row) > hdr.get("MAR", 3) else None
            screenshot = ""
            if "Screenshot" in hdr and len(row) > hdr["Screenshot"]:
                screenshot = row[hdr["Screenshot"]]

            events.append({"timestamp": ts, "event": event, "ear": ear, "mar": mar, "screenshot": screenshot})

    return events


def summarize_by_date(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Return counts per date and per event type.

    Result format: { '2025-11-14': {'Drowsiness': 3, 'Yawn': 1}, ... }
    """
    out = defaultdict(lambda: defaultdict(int))
    for e in events:
        date = e["timestamp"].date().isoformat()
        out[date][e["event"]] += 1
    return {d: dict(counts) for d, counts in out.items()}


def timeline_by_hour(events: List[Dict[str, Any]]) -> Dict[datetime.datetime, int]:
    """Aggregate event counts per hour.

    Returns a dict keyed by hour (datetime rounded to hour) -> count
    """
    counts = defaultdict(int)
    for e in events:
        ts = e["timestamp"].replace(minute=0, second=0, microsecond=0)
        counts[ts] += 1
    return dict(sorted(counts.items()))


def print_summary(events: List[Dict[str, Any]]):
    total = len(events)
    print(f"Total events: {total}")
    by_date = summarize_by_date(events)
    for date, counts in sorted(by_date.items()):
        parts = [f"{k}: {v}" for k, v in counts.items()]
        print(f"{date}: " + ", ".join(parts))


def plot_timeline(events: List[Dict[str, Any]]):
    """Plot a simple timeline (events per hour). Requires matplotlib."""
    if plt is None:
        raise RuntimeError("matplotlib is required to plot the timeline")

    counts = timeline_by_hour(events)
    if not counts:
        print("No events to plot")
        return

    xs = list(counts.keys())
    ys = [counts[k] for k in xs]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Time (hour)")
    plt.ylabel("Events")
    plt.title("Drowsiness events per hour")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_event_thumbnail(event: Dict[str, Any], max_size=(320, 240)):
    """Display the screenshot associated with an event (if available).

    Requires OpenCV and matplotlib to be installed. If the screenshot path
    is missing or not found, this function will print a message instead.
    """
    if plt is None or cv2 is None:
        raise RuntimeError("matplotlib and opencv-python are required to show thumbnails")

    path = event.get("screenshot", "")
    if not path or not os.path.exists(path):
        print("No screenshot available for this event.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Failed to read screenshot image.")
        return

    # Convert BGR->RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(max_size[0] / 100, max_size[1] / 100))
    plt.imshow(img)
    plt.axis("off")
    plt.title(event.get("event", ""))
    plt.show()


if __name__ == "__main__":
    log = "logs/drowsiness_log.csv"
    events = read_logs(log)
    print_summary(events)
    if plt is not None and events:
        try:
            plot_timeline(events)
        except Exception as exc:
            print("Plotting failed:", exc)
