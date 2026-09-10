"""MySQL persistence for processing runs, presence totals, events, and audit history."""
from __future__ import annotations
from datetime import datetime
from typing import Mapping
from app.core.presence import Presence, PresenceEvent


class MySQLPresenceStore:
    """Own a pipeline database connection and create its required schema on first use."""
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """Connect using PyMySQL and ensure the database tables and audit triggers exist."""
        try:
            import pymysql as mysql
        except ModuleNotFoundError as error:
            raise RuntimeError('Install pymysql to enable MySQL storage.') from error
        self._connector = mysql
        self._connection = mysql.connect(
            host=host, port=port, user=user, password=password, database=database,
            connect_timeout=5, read_timeout=10, write_timeout=10,
            init_command="SET time_zone='+00:00'",
        )
        try:
            self._create_tables()
            print(f'Connected to MySQL database {database} at {host}:{port}')
        except Exception:
            self._connection.close()
            raise

    def _create_tables(self):
        """Create missing tables and upgrade older installations with soft-delete support."""
        cursor = self._connection.cursor()
        cursor.execute('SHOW TABLES')
        existing_tables = {row[0] for row in cursor.fetchall()}
        if 'reid_runs' not in existing_tables:
            cursor.execute('''
                CREATE TABLE reid_runs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                source VARCHAR(512) NOT NULL,
                started_at DATETIME NOT NULL,
                completed_at DATETIME NOT NULL
                )
            ''')
        if 'person_presence' not in existing_tables:
            cursor.execute('''
                CREATE TABLE person_presence (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id BIGINT NOT NULL,
                person_name VARCHAR(255) NOT NULL,
                total_seconds DECIMAL(12, 3) NOT NULL,
                entries_count INT NOT NULL,
                exits_count INT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES reid_runs(id)
                )
            ''')
        if 'person_presence_events' not in existing_tables:
            cursor.execute('''
                CREATE TABLE person_presence_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id BIGINT NOT NULL,
                person_name VARCHAR(255) NOT NULL,
                event_type ENUM('IN', 'OUT') NOT NULL,
                occurred_at DATETIME NOT NULL,
                INDEX idx_presence_events_run_time (run_id, occurred_at)
                )
            ''')
        # ``deleted_at`` makes deletion reversible/auditable instead of removing rows outright.
        self._ensure_column(cursor, 'reid_runs', 'deleted_at', 'DATETIME NULL')
        self._ensure_column(cursor, 'person_presence', 'deleted_at', 'DATETIME NULL')
        self._ensure_column(cursor, 'person_presence_events', 'deleted_at', 'DATETIME NULL')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deleted_reid_runs (
                log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                original_id BIGINT NOT NULL,
                source VARCHAR(512) NOT NULL,
                started_at DATETIME NOT NULL,
                completed_at DATETIME NOT NULL,
                deleted_at DATETIME NOT NULL,
                archived_at DATETIME NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deleted_person_presence (
                log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                original_id BIGINT NOT NULL,
                run_id BIGINT NOT NULL,
                person_name VARCHAR(255) NOT NULL,
                total_seconds DECIMAL(12, 3) NOT NULL,
                entries_count INT NOT NULL,
                exits_count INT NOT NULL,
                deleted_at DATETIME NOT NULL,
                archived_at DATETIME NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deleted_person_presence_events (
                log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                original_id BIGINT NOT NULL,
                run_id BIGINT NOT NULL,
                person_name VARCHAR(255) NOT NULL,
                event_type ENUM('IN', 'OUT') NOT NULL,
                occurred_at DATETIME NOT NULL,
                deleted_at DATETIME NOT NULL,
                archived_at DATETIME NOT NULL
            )
        ''')
        self._create_audit_triggers(cursor)
        self._connection.commit()
        cursor.close()

    @staticmethod
    def _ensure_column(cursor, table: str, column: str, definition: str):
        """Add a column only when an existing installation does not already have it."""
        cursor.execute(
            'SELECT 1 FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s',
            (table, column),
        )
        if cursor.fetchone() is None:
            cursor.execute(f'ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}')

    @staticmethod
    def _create_audit_triggers(cursor):
        """Create one-time triggers that copy soft-deleted records into archive tables."""
        triggers = {
            'archive_deleted_reid_run': '''
                CREATE TRIGGER archive_deleted_reid_run AFTER UPDATE ON reid_runs
                FOR EACH ROW INSERT INTO deleted_reid_runs
                (original_id, source, started_at, completed_at, deleted_at, archived_at)
                SELECT NEW.id, NEW.source, NEW.started_at, NEW.completed_at, NEW.deleted_at, UTC_TIMESTAMP()
                WHERE OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
            ''',
            'archive_deleted_presence': '''
                CREATE TRIGGER archive_deleted_presence AFTER UPDATE ON person_presence
                FOR EACH ROW INSERT INTO deleted_person_presence
                (original_id, run_id, person_name, total_seconds, entries_count, exits_count, deleted_at, archived_at)
                SELECT NEW.id, NEW.run_id, NEW.person_name, NEW.total_seconds, NEW.entries_count,
                       NEW.exits_count, NEW.deleted_at, UTC_TIMESTAMP()
                WHERE OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
            ''',
            'archive_deleted_presence_event': '''
                CREATE TRIGGER archive_deleted_presence_event AFTER UPDATE ON person_presence_events
                FOR EACH ROW INSERT INTO deleted_person_presence_events
                (original_id, run_id, person_name, event_type, occurred_at, deleted_at, archived_at)
                SELECT NEW.id, NEW.run_id, NEW.person_name, NEW.event_type, NEW.occurred_at,
                       NEW.deleted_at, UTC_TIMESTAMP()
                WHERE OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
            ''',
        }
        for name, statement in triggers.items():
            cursor.execute(f'DROP TRIGGER IF EXISTS `{name}`')
            cursor.execute(statement)

    def save_run(self, source: str | int, started_at: datetime, completed_at: datetime,
                 records: Mapping[str, Presence], events: list[PresenceEvent]):
        """Save one run, its per-person totals, and its individual IN/OUT events atomically."""
        cursor = self._connection.cursor()
        cursor.execute(
            'INSERT INTO reid_runs (source, started_at, completed_at) VALUES (%s, %s, %s)',
            (str(source), started_at, completed_at),
        )
        run_id = cursor.lastrowid
        cursor.executemany(
            'INSERT INTO person_presence (run_id, person_name, total_seconds, entries_count, exits_count) VALUES (%s, %s, %s, %s, %s)',
            [(run_id, name, record.total_seconds, record.entries, record.exits)
             for name, record in records.items()],
        )
        if events:
            cursor.executemany(
                'INSERT INTO person_presence_events (run_id, person_name, event_type, occurred_at) VALUES (%s, %s, %s, %s)',
                [(run_id, event.person_name, event.event_type, event.occurred_at)
                 for event in events],
            )
        self._connection.commit()
        cursor.close()

    def close(self):
        """Close the owned database connection after the pipeline finishes."""
        self._connection.close()
