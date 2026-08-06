"""
Attendance tool using Playwright to scrape ERP and store records in PostgreSQL.

Security note: the target URL arrives from the caller, so every scrape is
validated by the SSRF guard before a browser is launched. Credentials passed in
are used only to fill the login form and are never logged or persisted.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import traceable
from app.services.url_guard import UnsafeURLError, assert_safe_outbound_url

logger = logging.getLogger(__name__)

# Headless Chromium is expensive; cap how many can run at once so a burst of
# scrape requests cannot exhaust server memory with concurrent browsers.
_MAX_CONCURRENT_BROWSERS = 2
_browser_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_BROWSERS)

_NAV_TIMEOUT_MS = 60_000
_VALID_STATUSES = {"present", "absent", "late"}


class AttendanceTool:
    """Scrape ERP attendance data and persist it."""

    @traceable(name="tool_attendance_scrape", run_type="tool", tags=["tool", "attendance"])
    async def scrape_and_store(
        self,
        user_id: str,
        erp_url: str,
        username: str,
        password: str,
        selectors: Dict[str, str],
    ) -> Dict[str, Any]:
        # Validated before anything is fetched. Raises UnsafeURLError, which the
        # route surfaces as a 400 rather than a server error.
        safe_url, hostname = assert_safe_outbound_url(erp_url)
        logger.info("Attendance scrape starting for user=%s host=%s", user_id, hostname)

        rows = await self._scrape_rows(
            erp_url=safe_url,
            username=username,
            password=password,
            selectors=selectors,
        )

        stored_ids: List[str] = []
        skipped: List[Dict[str, str]] = []

        for row in rows:
            row_date = self._parse_date(row.get("date"))
            if row_date is None:
                # Reported rather than silently dropped: a layout change that
                # breaks date parsing otherwise looks like an empty semester.
                skipped.append({
                    "reason": "unparseable_date",
                    "raw_date": str(row.get("date", ""))[:50],
                    "subject": str(row.get("subject", ""))[:80],
                })
                continue

            subject = (row.get("subject") or "Unknown Subject").strip()
            status = (row.get("status") or "absent").strip().lower()
            notes = (row.get("notes") or "").strip() or None

            if status not in _VALID_STATUSES:
                status = "absent"

            # Idempotent on (user_id, date, subject): re-running a scrape
            # updates existing rows instead of duplicating them, which would
            # inflate the totals used for attendance-percentage warnings.
            rec_id = await memory_manager.store_attendance(
                user_id=user_id,
                date=row_date,
                subject=subject,
                status=status,
                notes=notes,
            )
            stored_ids.append(rec_id)

        if skipped:
            logger.warning(
                "Attendance scrape for user=%s skipped %d/%d rows (first reason: %s)",
                user_id, len(skipped), len(rows), skipped[0]["reason"],
            )

        return {
            "tool": "attendance_scraper",
            "success": True,
            "user_id": user_id,
            "scraped_count": len(rows),
            "stored_count": len(stored_ids),
            "skipped_count": len(skipped),
            "skipped": skipped[:10],
            "stored_ids": stored_ids,
        }

    async def _scrape_rows(
        self,
        erp_url: str,
        username: str,
        password: str,
        selectors: Dict[str, str],
    ) -> List[Dict[str, str]]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error(
                "Playwright is not installed — attendance scraping is unavailable. "
                "Install it with: pip install playwright && playwright install chromium"
            )
            return []

        login_user = selectors.get("username_input", "input[name='username']")
        login_pass = selectors.get("password_input", "input[name='password']")
        login_button = selectors.get("login_button", "button[type='submit']")
        attendance_link = selectors.get("attendance_nav", "a[href*='attendance']")
        rows_selector = selectors.get("rows", "table tbody tr")

        extracted: List[Dict[str, str]] = []

        async with _browser_semaphore:
            async with async_playwright() as p:
                browser = None
                try:
                    # launch() lives inside the try so a failure anywhere after
                    # it — including new_page() — still reaches the finally
                    # block and closes the browser process.
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()

                    await page.goto(erp_url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
                    await page.fill(login_user, username)
                    await page.fill(login_pass, password)
                    await page.click(login_button)
                    await page.wait_for_load_state("networkidle", timeout=_NAV_TIMEOUT_MS)

                    await page.click(attendance_link)
                    await page.wait_for_load_state("networkidle", timeout=_NAV_TIMEOUT_MS)

                    row_elements = await page.query_selector_all(rows_selector)
                    for row in row_elements:
                        cells = await row.query_selector_all("td")
                        if len(cells) < 3:
                            continue
                        values = [((await c.inner_text()) or "").strip() for c in cells]

                        extracted.append(
                            {
                                "date": values[0] if len(values) > 0 else "",
                                "subject": values[1] if len(values) > 1 else "",
                                "status": values[2] if len(values) > 2 else "",
                                "notes": values[3] if len(values) > 3 else "",
                            }
                        )
                except Exception as exc:
                    # Log the exception type and message only — never the page
                    # content or the credentials that were submitted to it.
                    logger.error(
                        "ERP scrape failed (%s): %s", type(exc).__name__, exc
                    )
                    raise
                finally:
                    if browser is not None:
                        try:
                            await browser.close()
                        except Exception as close_exc:
                            logger.warning("Failed to close browser cleanly: %s", close_exc)

        return extracted

    def _parse_date(self, raw_date: Optional[str]):
        if not raw_date:
            return None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw_date.strip(), fmt).date()
            except ValueError:
                continue
        return None


attendance_tool = AttendanceTool()
