"""Data-only recovery of a stopped, identified PySR ordinary-expression run."""
import csv
import io
import math
import re
import uuid
from pathlib import Path
from importlib.metadata import version

from ..data import digest, read_json, write_json


def begin_run(directory):
    directory = Path(directory)
    job_path = directory / 'job.json'
    run_id = uuid.uuid4().hex
    if job_path.is_file():
        job = read_json(job_path)
        marker = directory / 'pysr_run.json'
        if marker.exists():
            raise ValueError('PySR run marker already exists; use a new run directory')
        write_json(marker, {'run_id': run_id, 'job_sha256': digest(job_path),
                           'train_sha256': digest(job['train']), 'version': version('pysr'),
                           'adapter_sha256': digest(Path(__file__).with_name('pysr.py'))})
    return run_id


def validate_csv(content):
    if not content or len(content) > 20_000_000 or not content.endswith(b'\n'):
        raise ValueError('Incomplete or oversized PySR hall-of-fame CSV')
    reader = csv.DictReader(io.StringIO(content.decode('utf-8')), strict=True)
    if reader.fieldnames != ['Complexity', 'Loss', 'Equation']:
        raise ValueError('PySR CSV requires exactly three unique ordered columns')
    rows = list(reader)
    if not rows or len(rows) > 10000:
        raise ValueError('Empty or oversized PySR hall of fame')
    complexities = set()
    for row in rows:
        if set(row) != {'Complexity', 'Loss', 'Equation'} or any(v is None for v in row.values()):
            raise ValueError('Invalid PySR CSV schema or truncated row')
        complexity = int(row['Complexity']); loss = float(row['Loss'])
        if complexity < 1 or complexity in complexities or math.isnan(loss) or loss < 0 or not row['Equation'].strip():
            raise ValueError('Invalid PySR hall-of-fame record')
        complexities.add(complexity)
    return len(rows)


def recovered_run(directory):
    """Called only in a fresh recovery worker after the writer group is reaped.

    Require identical primary/backup generations: do not guess which differing
    generation is complete. Structural checks + native prediction parity do not
    certify arbitrary external tampering or optimizer state.
    """
    directory = Path(directory).resolve()
    marker = read_json(directory / 'pysr_run.json'); job_path = directory / 'job.json'
    job = read_json(job_path)
    if (marker['job_sha256'] != digest(job_path) or marker['train_sha256'] != digest(job['train'])
            or marker.get('adapter_sha256') != digest(Path(__file__).with_name('pysr.py'))
            or marker['version'] != version('pysr') or marker['version'] != '2.7.2'):
        raise ValueError('PySR recovery run/job/Train/version mismatch')
    run_id = marker['run_id']
    if not isinstance(run_id, str) or not re.fullmatch('[0-9a-f]{32}', run_id):
        raise ValueError('Invalid PySR run identity')
    run = directory / 'native' / run_id
    if run.resolve() != run or run.is_symlink():
        raise ValueError('PySR run path escapes its recorded directory')
    primary = run / 'hall_of_fame.csv'; backup = run / 'hall_of_fame.csv.bak'
    for p in [primary, backup]:
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 20_000_000:
            raise ValueError('Missing, redirected or oversized PySR CSV generation')
    content = primary.read_bytes()
    rows = validate_csv(content)
    validate_csv(backup.read_bytes())
    if content != backup.read_bytes():
        raise ValueError('PySR CSV generations differ; recovery refuses an ambiguous/incomplete generation')
    snapshot = directory / 'recovery_csv'
    snapshot.mkdir()  # Fresh destination: no stale pickle/cache or previous attempt.
    (snapshot / 'hall_of_fame.csv').write_bytes(content)
    evidence = {'run_id': run_id, 'csv_sha256': digest(primary), 'rows': rows,
                'source': str(primary), 'snapshot': str(snapshot),
                'generation': 'identical complete CSV and backup after writer stopped',
                'scope': 'ordinary expressions only; prediction parity rechecked; no convergence or optimizer resume'}
    write_json(directory / 'pysr_recovery_evidence.json', evidence)
    return snapshot, evidence
