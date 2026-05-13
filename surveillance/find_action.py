import logging

from surveillance import config
from surveillance.db import ActivityDB

logger = logging.getLogger(__name__)


def run_find(class_name: str):
    db = ActivityDB(config.ACTIVITY_DB_PATH)
    row = db.find_last_seen(class_name)
    db.close()

    if not row:
        print(f"'{class_name}' not found in activity records.")
        return 1

    print(f"{class_name}: last seen in {row['room_name']} at {row['datetime_to']}")
    return 0
