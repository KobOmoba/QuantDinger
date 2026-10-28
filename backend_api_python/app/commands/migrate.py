"""Database migration entrypoint for deployments with connection retry."""

from __future__ import annotations

import time


def main() -> None:
    print("[migration] database bootstrap starting", flush=True)
    from app.utils.db import init_database

    max_retries = 10
    retry_interval = 3
    for attempt in range(1, max_retries + 1):
        try:
            init_database(strict_migrations=True)
            print("[migration] database bootstrap finished", flush=True)
            return
        except Exception as exc:
            if attempt < max_retries:
                print(
                    f"[migration] Connection attempt {attempt}/{max_retries} failed ({exc}). "
                    f"Retrying in {retry_interval}s...",
                    flush=True,
                )
                time.sleep(retry_interval)
            else:
                print(
                    f"[migration] All {max_retries} attempts failed.",
                    flush=True,
                )
                raise


if __name__ == "__main__":
    main()
