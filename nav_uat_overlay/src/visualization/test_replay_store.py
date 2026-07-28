import json
import tempfile
import unittest
from pathlib import Path

from visualization.replay_store import ReplayStore


class ReplayStoreTest(unittest.TestCase):
    def test_list_tasks_skips_invalid_metadata_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ReplayStore(history_dir=temp_dir)

            valid_dir = Path(temp_dir) / "valid-task"
            valid_dir.mkdir(parents=True, exist_ok=True)
            (valid_dir / "task_meta.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "goal_text": "door",
                        "status": "completed",
                        "started_at": 1.0,
                        "ended_at": 2.0,
                    }
                ),
                encoding="utf-8",
            )

            invalid_dir = Path(temp_dir) / "invalid-task"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            (invalid_dir / "task_meta.json").write_text("", encoding="utf-8")
            (invalid_dir / "snapshot.json").write_text("", encoding="utf-8")

            tasks = store.list_tasks()

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], "task-1")
            self.assertTrue(store.get_task("task-1") is not None)
            self.assertIsNone(store.get_task("missing-task"))

    def test_load_events_skips_invalid_json_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ReplayStore(history_dir=temp_dir)
            task_id = "task-2"
            task_dir = Path(temp_dir) / "task-dir"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task_meta.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "goal_text": "desk",
                        "status": "completed",
                        "started_at": 1.0,
                        "ended_at": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "event", "event_name": "task_received"}),
                        "{bad json",
                        json.dumps({"type": "event", "event_name": "task_completed"}),
                    ]
                ),
                encoding="utf-8",
            )

            events = store.load_events(task_id)

            self.assertEqual(
                [event["event_name"] for event in events],
                ["task_received", "task_completed"],
            )


if __name__ == "__main__":
    unittest.main()
