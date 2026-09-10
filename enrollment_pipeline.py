#!/usr/bin/env python
"""Orchestrate interactive enrollment followed by gallery construction."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from app.config import Settings
from app.utils.pipeline_logging import configure_pipeline_logging, get_pipeline_logger


def run_stage(stage_name: str, module: str, logger) -> None:
    """Run one existing script as an isolated Python stage."""
    logger.info('Starting enrollment stage: %s', stage_name)
    try:
        subprocess.run(
            [sys.executable, '-m', module],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        logger.error('Enrollment stage failed: %s (exit code %s)', stage_name, error.returncode)
        raise RuntimeError(f'{stage_name} failed with exit code {error.returncode}.') from error
    logger.info('Completed enrollment stage: %s', stage_name)


def main() -> int:
    """Run enrollment and gallery construction in sequence."""
    parser = argparse.ArgumentParser(description='Enroll a person and build the ReID gallery.')
    parser.add_argument(
        '--build-only',
        action='store_true',
        help='Skip camera enrollment and rebuild embeddings from existing images.',
    )
    args = parser.parse_args()

    settings = Settings()
    log_path = configure_pipeline_logging(settings.output_dir)
    logger = get_pipeline_logger()
    logger.info('Starting enrollment pipeline: build_only=%s, gallery=%s', args.build_only, settings.gallery_dir)

    try:
        if not args.build_only:
            run_stage('camera enrollment', 'app.scripts.enroll', logger)
        run_stage('gallery construction', 'app.scripts.build_gallery', logger)
    except KeyboardInterrupt:
        logger.warning('Enrollment pipeline stopped by user.')
        print('Enrollment pipeline stopped by user.', file=sys.stderr)
        return 130
    except Exception as error:
        logger.exception('Enrollment pipeline failed.')
        print(f'Enrollment pipeline failed: {error}', file=sys.stderr)
        print(f'See {log_path} for details.', file=sys.stderr)
        return 1

    logger.info('Enrollment pipeline completed successfully.')
    print(f'Enrollment pipeline completed successfully. Log: {log_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
