#!/usr/bin/env python
"""Command-line entry point for processing a webcam, video file, or stream."""
import sys
from pathlib import Path
from app.config import Settings
from app.core.pipeline import ReIDPipeline
from app.utils.pipeline_logging import configure_pipeline_logging, get_pipeline_logger


def resolve_source(source_arg: str, project_root: Path) -> str:
    """Resolve a CLI source while preserving URL and camera semantics.

    Local paths are rooted at the repository so launching from another working
    directory still finds the media. A path without an extension may identify
    one unique video prefix, but ambiguous prefixes fail explicitly.
    """
    # Network streams are valid OpenCV sources but are not filesystem paths.
    if '://' in source_arg:
        return source_arg
    candidate = Path(source_arg).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    if candidate.is_file():
        # Return the canonical absolute path so logs and downstream errors name
        # the exact file that was selected.
        return str(candidate)

    parent = candidate.parent if candidate.parent.exists() else project_root
    prefix = candidate.name if candidate.parent.exists() else source_arg
    matches = sorted(path for path in parent.glob(f'{prefix}*')
                     if path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'})
    if len(matches) == 1:
        # Prefix resolution is safe only when exactly one supported video
        # matches; silently choosing among several recordings is dangerous.
        return str(matches[0])
    if len(matches) > 1:
        names = ', '.join(path.name for path in matches)
        raise FileNotFoundError(f'Source prefix is ambiguous: {source_arg}. Matches: {names}')
    raise FileNotFoundError(f'Source video does not exist: {candidate}')


if __name__ == '__main__':
    try:
        settings = Settings()
        log_path = configure_pipeline_logging(settings.output_dir)
        logger = get_pipeline_logger()

        # Camera indexes are integers (0 is normally the default camera); other values are paths/URLs.
        source_arg = sys.argv[1] if len(sys.argv) > 1 else '0'
        source = int(source_arg) if source_arg.isdecimal() else resolve_source(source_arg, settings.project_root)
        output = settings.output_dir / 'output_video.mp4'
        logger.info('Starting pipeline command: source=%s, log=%s', source, log_path)
        pipeline = ReIDPipeline(settings)
        pipeline.run_on_video(source, save_to_db=True, output_video=output)
        print(f'Pipeline completed successfully. Log: {log_path}')
    except KeyboardInterrupt:
        print('Pipeline stopped by user.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        logger = get_pipeline_logger()
        logger.exception('Pipeline command failed.')
        print(f'Pipeline failed: {error}', file=sys.stderr)
        print('See output/pipeline.log for details.', file=sys.stderr)
        sys.exit(1)
