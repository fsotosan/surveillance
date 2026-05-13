import logging

from surveillance import config
from surveillance.db import ActivityDB

logger = logging.getLogger(__name__)


def run_report(dt_from: str, dt_to: str, class_filter: str | None = None):
    db = ActivityDB(config.ACTIVITY_DB_PATH)
    rows = db.query_activity(dt_from, dt_to, class_filter)
    db.close()

    if not rows:
        print("No activity found in the given range.")
        return 0

    header = f"{'Class':<14} {'Room':<14} {'Cam':<6} {'From':<20} {'To':<20}"
    sep = '-' * len(header)
    print(f"Activity from {dt_from} to {dt_to}")
    if class_filter:
        print(f"Filter: {class_filter}")
    print()
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['class_name']:<14} {r['room_name']:<14} {r['camera_number']:<6} "
              f"{r['datetime_from']:<20} {r['datetime_to']:<20}")
    return 0
