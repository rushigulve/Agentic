"""
pipeline/scheduler.py

APScheduler-based periodic pipeline execution.
Runs the full ingestion pipeline on a configurable interval.
"""

import asyncio
import signal
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from config import FETCH_INTERVAL_MINUTES

logger = logging.getLogger(__name__)

_scheduler = None


def start_scheduler(pipeline_func):
    """
    Start the APScheduler to run the pipeline at regular intervals.

    Args:
        pipeline_func: async function to run (should be run_once from run_pipeline)
    """
    global _scheduler

    _scheduler = AsyncIOScheduler()

    _scheduler.add_job(
        pipeline_func,
        "interval",
        minutes=FETCH_INTERVAL_MINUTES,
        id="news_pipeline",
        name="News Ingestion Pipeline",
        max_instances=1,              # prevent overlapping runs
        misfire_grace_time=60,        # allow 60s late start
    )

    # also run immediately on start
    _scheduler.add_job(
        pipeline_func,
        id="news_pipeline_initial",
        name="Initial Pipeline Run",
    )

    _scheduler.start()
    logger.info(
        f"[scheduler] Started — running every {FETCH_INTERVAL_MINUTES} minutes"
    )

    return _scheduler


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("[scheduler] Stopped")


async def run_scheduled(pipeline_func):
    """
    Start scheduler and keep running until interrupted.

    Args:
        pipeline_func: async function to run periodically
    """
    loop = asyncio.get_event_loop()

    # handle graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("\n[scheduler] Shutdown signal received...")
        shutdown_event.set()

    # register signal handlers (works on Windows too)
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        # KeyboardInterrupt will be caught in the except block below
        pass

    start_scheduler(pipeline_func)

    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop_scheduler()
