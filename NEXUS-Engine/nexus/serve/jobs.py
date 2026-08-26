"""Менеджер фоновых задач обучения (запуск в отдельном процессе).

Каждая задача — вызов CLI (`python -m nexus.cli ...`) с логом в файл, статусом
и кодом возврата. Это позволяет запускать обучение из API, не блокируя
инференс, и переживать долгие прогоны.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    id: str
    command: List[str]
    status: str = "pending"          # pending | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    returncode: Optional[int] = None
    log_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = round((self.finished or time.time()) - self.created, 1)
        return d


class JobManager:
    def __init__(self, log_dir: str = "artifacts/jobs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def submit(self, args: List[str]) -> Job:
        job_id = uuid.uuid4().hex[:12]
        command = [sys.executable, "-m", "nexus.cli", *args]
        log_path = os.path.join(self.log_dir, f"{job_id}.log")
        job = Job(job_id, command, log_path=log_path)
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        job.status = "running"
        with open(job.log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(shlex.quote(c) for c in job.command)}\n")
            log.flush()
            try:
                proc = subprocess.Popen(job.command, stdout=log, stderr=subprocess.STDOUT)
                with self._lock:
                    self._procs[job.id] = proc
                job.returncode = proc.wait()
            except Exception as exc:  # pragma: no cover
                log.write(f"\n[ошибка запуска] {exc}\n")
                job.returncode = -1
        job.finished = time.time()
        job.status = "done" if job.returncode == 0 else "failed"
        with self._lock:
            self._procs.pop(job.id, None)

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._lock:
            proc = self._procs.get(job_id)
            job = self._jobs.get(job_id)
        if proc and job:
            proc.terminate()
            job.status = "cancelled"
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def logs(self, job_id: str, tail: int = 100) -> str:
        job = self._jobs.get(job_id)
        if not job or not os.path.exists(job.log_path):
            return ""
        with open(job.log_path, encoding="utf-8", errors="ignore") as fh:
            return "".join(fh.readlines()[-tail:])

    def list(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created)]
