"""TaskManager.wait_for_process: blocking helper used by scripts/harnesses.

Regression: the helper used to call ``Task.wait()`` straight away, which raised
``AttributeError`` for a task the compute worker had not started yet (hit while
driving a real-machine training run from a foreground script).
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest

from mikazuki.tasks import TaskManager, TaskStatus


class WaitForProcessTests(unittest.TestCase):
    # No persist_path: the worker's trailing _persist() would otherwise write
    # into the temp dir after wait_for_process() returns and race its cleanup.
    def test_returns_when_queued_task_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = TaskManager(max_concurrent=1)
            task = manager.create_task(
                [sys.executable, "-c", "raise SystemExit(0)"],
                os.environ.copy(),
                metadata={"job_label": "wait-for-process smoke"},
                cwd=tmp,
            )
            manager.submit(task)
            manager.wait_for_process(task.task_id)
            self.assertIn(task.status, (TaskStatus.FINISHED, TaskStatus.FAILED))
            self.assertEqual(task.returncode, 0)

    def test_unknown_task_id_does_not_hang(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = TaskManager(max_concurrent=1)
            returned = threading.Event()

            def run() -> None:
                manager.wait_for_process("missing-task")
                returned.set()

            threading.Thread(target=run, daemon=True).start()
            self.assertTrue(returned.wait(5), "wait_for_process hung on an unknown task id")


if __name__ == "__main__":
    unittest.main()
