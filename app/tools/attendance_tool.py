"""
Attendance tool using Playwright to scrape ERP and store records in PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import traceable


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
        rows = await self._scrape_rows(
            erp_url=erp_url,
            username=username,
            password=password,
            selectors=selectors,
        )

        stored_ids: List[str] = []
        for row in rows:
            row_date = self._parse_date(row.get("date"))
            if row_date is None:
                continue

            subject = (row.get("subject") or "Unknown Subject").strip()
            status = (row.get("status") or "absent").strip().lower()
            notes = (row.get("notes") or "").strip() or None

            if status not in {"present", "absent", "late"}:
                status = "absent"

            rec_id = await memory_manager.store_attendance(
                user_id=user_id,
                date=row_date,
                subject=subject,
                status=status,
                notes=notes,
            )
            stored_ids.append(rec_id)

        return {
            "tool": "attendance_scraper",
            "success": True,
            "user_id": user_id,
            "scraped_count": len(rows),
            "stored_count": len(stored_ids),
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
        except Exception:
            return []

        login_user = selectors.get("username_input", "input[name='username']")
        login_pass = selectors.get("password_input", "input[name='password']")
        login_button = selectors.get("login_button", "button[type='submit']")
        attendance_link = selectors.get("attendance_nav", "a[href*='attendance']")
        rows_selector = selectors.get("rows", "table tbody tr")

        extracted: List[Dict[str, str]] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(erp_url, wait_until="domcontentloaded", timeout=60000)
                await page.fill(login_user, username)
                await page.fill(login_pass, password)
                await page.click(login_button)
                await page.wait_for_load_state("networkidle", timeout=60000)

                await page.click(attendance_link)
                await page.wait_for_load_state("networkidle", timeout=60000)

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
            finally:
                await browser.close()

        return extracted

    def _parse_date(self, raw_date: str):
        if not raw_date:
            return None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw_date.strip(), fmt).date()
            except Exception:
                continue
        return None


attendance_tool = AttendanceTool()
