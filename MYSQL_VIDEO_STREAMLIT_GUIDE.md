# Saving processed videos in MySQL and displaying them in Streamlit

This project already creates an annotated output video in `output/output_video.mp4` and saves each run's presence data in MySQL. This guide adds one video to each `reid_runs` record so the **Analytics** page can play the exact processed video for a selected run.

> Important: Storing videos inside MySQL works well for short demonstrations and small recordings. For long or frequent recordings, store the MP4 in object storage (S3, Cloudflare R2, Azure Blob Storage, etc.) and keep only its URL in MySQL. That is cheaper, faster, and avoids MySQL packet-size limits.

## 1. Add video columns to `reid_runs`

Use `LONGBLOB` because an MP4 is binary data. Keep the filename and MIME type so the dashboard knows how to present it.

```sql
ALTER TABLE reid_runs
    ADD COLUMN output_video LONGBLOB NULL,
    ADD COLUMN output_video_name VARCHAR(255) NULL,
    ADD COLUMN output_video_mime VARCHAR(100) NULL;
```

For a fresh installation, put these columns directly into the `CREATE TABLE reid_runs` statement in `app/storage/mysql_store.py`. For an existing installation, the project's `_ensure_column` helper can add them automatically:

```python
self._ensure_column(cursor, 'reid_runs', 'output_video', 'LONGBLOB NULL')
self._ensure_column(cursor, 'reid_runs', 'output_video_name', 'VARCHAR(255) NULL')
self._ensure_column(cursor, 'reid_runs', 'output_video_mime', 'VARCHAR(100) NULL')
```

### MySQL configuration limit

MySQL rejects a single request larger than `max_allowed_packet`. Check it with:

```sql
SHOW VARIABLES LIKE 'max_allowed_packet';
```

For example, an 80 MB MP4 needs a server packet setting larger than 80 MB. On a server you manage, add this to MySQL's configuration and restart MySQL:

```ini
[mysqld]
max_allowed_packet=128M
```

Also set an application-level maximum upload size. Do not let a webcam run create unlimited database rows or blobs.

## 2. Save the video bytes with the run

Change `MySQLPresenceStore.save_run` to accept `output_video: Path | None`. Read the file *after* OpenCV releases the `VideoWriter`; otherwise the MP4 may be incomplete.

```python
from pathlib import Path

def save_run(
    self,
    source: str | int,
    started_at: datetime,
    completed_at: datetime,
    records: Mapping[str, Presence],
    events: list[PresenceEvent],
    output_video: Path | None = None,
) -> None:
    video_bytes = None
    video_name = None
    video_mime = None

    if output_video is not None and output_video.exists():
        video_bytes = output_video.read_bytes()
        video_name = output_video.name
        video_mime = 'video/mp4'

    cursor = self._connection.cursor()
    cursor.execute(
        '''
        INSERT INTO reid_runs
            (source, started_at, completed_at,
             output_video, output_video_name, output_video_mime)
        VALUES (%s, %s, %s, %s, %s, %s)
        ''',
        (str(source), started_at, completed_at,
         video_bytes, video_name, video_mime),
    )
    run_id = cursor.lastrowid
    # Keep the existing person_presence and event inserts here.
    self._connection.commit()
    cursor.close()
```

`pymysql` safely sends the `bytes` value as a parameter. Do not build a SQL string with video bytes yourself.

## 3. Pass the video path from the pipeline

In `app/core/pipeline.py`, the `finally` block releases `out` before saving the run. Pass the existing `output_video` argument after that release:

```python
self.store.save_run(
    source=str(source),
    started_at=started_at,
    completed_at=completed_at,
    records=self.presence.records,
    events=self.presence.events,
    output_video=output_video,
)
```

The runner already supplies an output path:

```python
output = settings.output_dir / 'output_video.mp4'
pipeline.run_on_video(source, save_to_db=True, output_video=output)
```

### Avoid overwriting earlier videos

The current fixed name, `output_video.mp4`, is fine because the bytes are saved before the next run. A timestamped name is clearer while debugging:

```python
output = settings.output_dir / f'run_{datetime.now():%Y%m%d_%H%M%S}.mp4'
```

## 4. Read the video for a selected run

Add this function in `app/dashboard/shared.py`:

```python
@st.cache_data(ttl=3, show_spinner=False)
def load_run_video(run_id: int) -> tuple[bytes | None, str]:
    """Return the saved MP4 bytes and MIME type for one non-archived run."""
    db = connection()
    try:
        with db.cursor() as cursor:
            cursor.execute(
                '''
                SELECT output_video, COALESCE(output_video_mime, 'video/mp4') AS mime
                FROM reid_runs
                WHERE id=%s AND deleted_at IS NULL
                ''',
                (run_id,),
            )
            row = cursor.fetchone()
            return (row['output_video'], row['mime']) if row else (None, 'video/mp4')
    finally:
        db.close()
```

Do **not** include `output_video` in `load_runs()`: its list query should stay small and fast. Fetch it only after the user chooses one run.

## 5. Play the video in the Analytics page

In `app/dashboard/analytics.py`, import the new helper and add this section after the selected run is known:

```python
from app.dashboard.shared import apply_theme, load_run_details, load_run_video, load_runs

# ... existing selected-run code ...

st.markdown("<div class='section-title'>Processed video</div>", unsafe_allow_html=True)
video_bytes, mime = load_run_video(run['id'])
if video_bytes:
    st.video(video_bytes, format=mime)
    st.download_button(
        'Download processed video',
        data=video_bytes,
        file_name=f"run_{run['id']}.mp4",
        mime=mime,
    )
else:
    st.info('No processed video was saved for this run.')
```

`st.video` accepts bytes, so the browser can play the BLOB without creating a public temporary URL.

## 6. Keep archive behavior consistent

The dashboard already uses a soft delete (`deleted_at`). Because the video belongs to `reid_runs`, it is automatically hidden by the `WHERE deleted_at IS NULL` condition. The existing audit trigger should be extended if archived videos also need to be retained:

```sql
ALTER TABLE deleted_reid_runs ADD COLUMN output_video LONGBLOB NULL;
ALTER TABLE deleted_reid_runs ADD COLUMN output_video_name VARCHAR(255) NULL;
ALTER TABLE deleted_reid_runs ADD COLUMN output_video_mime VARCHAR(100) NULL;
```

Then include those columns in the `archive_deleted_reid_run` trigger. If archival video retention is not required, omit the BLOB from `deleted_reid_runs` to avoid duplicating a potentially large file.

## Recommended design for production

For a deployed app, replace the BLOB with two metadata columns:

```sql
ALTER TABLE reid_runs
    ADD COLUMN output_video_url VARCHAR(2048) NULL,
    ADD COLUMN output_video_mime VARCHAR(100) NULL;
```

Upload the finalized MP4 to private object storage, store a signed/temporary URL, and pass that URL to `st.video`. This keeps MySQL focused on relational data and lets the storage provider serve video efficiently.

## Test checklist

1. Process a short, 5-10 second camera or test video.
2. Confirm `reid_runs.output_video` is not `NULL` and has a sensible byte count.
3. Select that run in Analytics and verify playback and download.
4. Archive the run and confirm it no longer appears in Analytics.
5. Test a video near the configured limit to confirm MySQL's packet setting and Streamlit memory are sufficient.

