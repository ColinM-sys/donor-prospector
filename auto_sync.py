"""
Donor Prospector — Auto Sync Pipeline
Reads sync_config.json and pulls all configured data sources into ChromaDB.
Can be run as a one-shot or on a schedule.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime

from ingest import (
    ingest_irs990,
    fetch_cms_payments,
    scrape_url,
    store_records,
    get_collection_stats,
)

CONFIG_PATH = Path(__file__).parent / "sync_config.json"
LOG_PATH = Path(__file__).parent / "data" / "sync_log.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def run_sync(progress_callback=None) -> dict:
    """Run a full data sync from all configured sources. Returns stats."""
    config = load_config()
    stats = {
        "started_at": datetime.now().isoformat(),
        "irs990_orgs": 0,
        "irs990_chunks": 0,
        "cms_records": 0,
        "cms_chunks": 0,
        "web_pages": 0,
        "web_chunks": 0,
        "errors": [],
    }

    total_steps = len(config.get("irs990_searches", [])) + len(config.get("cms_searches", [])) + len(config.get("web_urls", []))
    step = 0

    # IRS 990
    for search in config.get("irs990_searches", []):
        step += 1
        if progress_callback:
            progress_callback(step / total_steps, f"IRS 990: {search['query']}...")
        try:
            recs = ingest_irs990(search["query"], state=search.get("state", ""), sector=search.get("sector", ""))
            n = store_records(recs)
            stats["irs990_orgs"] += len(recs)
            stats["irs990_chunks"] += n
        except Exception as e:
            stats["errors"].append(f"IRS 990 '{search['query']}': {str(e)}")

    # CMS Open Payments
    for search in config.get("cms_searches", []):
        step += 1
        if progress_callback:
            progress_callback(step / total_steps, f"CMS: {search.get('company', search.get('physician_name', ''))}...")
        try:
            recs = fetch_cms_payments(
                physician_name=search.get("physician_name", ""),
                company=search.get("company", ""),
                state=search.get("state", ""),
            )
            n = store_records(recs)
            stats["cms_records"] += len(recs)
            stats["cms_chunks"] += n
        except Exception as e:
            stats["errors"].append(f"CMS '{search}': {str(e)}")

    # Web URLs
    for entry in config.get("web_urls", []):
        step += 1
        url = entry if isinstance(entry, str) else entry.get("url", "")
        terms = entry.get("terms", []) if isinstance(entry, dict) else None
        if not url:
            continue
        if progress_callback:
            progress_callback(step / total_steps, f"Web: {url[:50]}...")
        try:
            recs = scrape_url(url, terms=terms)
            n = store_records(recs)
            stats["web_pages"] += 1
            stats["web_chunks"] += n
        except Exception as e:
            stats["errors"].append(f"Web '{url}': {str(e)}")

    stats["finished_at"] = datetime.now().isoformat()
    stats["total_db_chunks"] = get_collection_stats()["total_chunks"]

    # Save log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append(stats)
    # Keep last 50 syncs
    log = log[-50:]
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return stats


def add_irs990_search(query: str, state: str = "", sector: str = ""):
    config = load_config()
    config.setdefault("irs990_searches", []).append({"query": query, "state": state, "sector": sector})
    save_config(config)


def add_cms_search(physician_name: str = "", company: str = "", state: str = ""):
    config = load_config()
    entry = {}
    if physician_name:
        entry["physician_name"] = physician_name
    if company:
        entry["company"] = company
    if state:
        entry["state"] = state
    config.setdefault("cms_searches", []).append(entry)
    save_config(config)


def add_web_url(url: str, terms: list[str] | None = None):
    config = load_config()
    entry = {"url": url}
    if terms:
        entry["terms"] = terms
    config.setdefault("web_urls", []).append(entry)
    save_config(config)


def remove_search(source: str, index: int):
    config = load_config()
    key = {"irs990": "irs990_searches", "cms": "cms_searches", "web": "web_urls"}.get(source)
    if key and 0 <= index < len(config.get(key, [])):
        config[key].pop(index)
        save_config(config)


class SyncScheduler:
    """Background scheduler that runs sync at configured intervals."""

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self, interval_hours: float = None):
        if self._running:
            return
        config = load_config()
        interval = interval_hours or config.get("sync_interval_hours", 24)
        self._running = True

        def loop():
            while self._running:
                print(f"[{datetime.now().isoformat()}] Running auto-sync...")
                try:
                    stats = run_sync()
                    total = stats["irs990_orgs"] + stats["cms_records"]
                    print(f"  Synced: {stats['irs990_orgs']} orgs, {stats['cms_records']} payments, {stats['web_pages']} pages. DB: {stats['total_db_chunks']} chunks.")
                    if stats["errors"]:
                        print(f"  Errors: {len(stats['errors'])}")
                except Exception as e:
                    print(f"  Sync error: {e}")
                time.sleep(interval * 3600)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


if __name__ == "__main__":
    import sys
    if "--schedule" in sys.argv:
        print("Starting scheduled sync...")
        sched = SyncScheduler()
        sched.start()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            sched.stop()
    else:
        print("Running one-time sync...")
        stats = run_sync(progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"))
        print(f"\nSync complete:")
        print(f"  IRS 990: {stats['irs990_orgs']} orgs ({stats['irs990_chunks']} chunks)")
        print(f"  CMS: {stats['cms_records']} payments ({stats['cms_chunks']} chunks)")
        print(f"  Web: {stats['web_pages']} pages ({stats['web_chunks']} chunks)")
        print(f"  Total DB: {stats['total_db_chunks']} chunks")
        if stats["errors"]:
            print(f"  Errors ({len(stats['errors'])}):")
            for e in stats["errors"]:
                print(f"    - {e}")
