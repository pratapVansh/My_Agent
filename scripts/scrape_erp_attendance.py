"""
Scrape attendance from RGIPT ERP portal (rgipterp.com).

Handles the text-based captcha automatically using Playwright.
Stores results in the agent's attendance database so the academic
agent can report accurate percentages.

Usage:
  python scripts/scrape_erp_attendance.py --user-id vansh --rollno 23IT3048 --password YOUR_PASSWORD

Options:
  --headless        Run browser invisibly (default: visible so you can watch)
  --replace         Clear previous ERP records before importing (default: True)
  --dry-run         Print scraped data without saving to database
"""

import asyncio
import sys
import argparse
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sure app modules are importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Selectors for rgipterp.com ────────────────────────────────────────────────
# Discovered from page inspection. If the ERP changes layout, update these.

LOGIN_URL = "https://rgipterp.com/erp/login.php"

SELECTORS = {
    # Login page
    "rollno_input":      "input[type='text']",           # first text field
    "password_input":    "input[type='password']",
    "captcha_display":   ".captcha-code",                # element showing the code
    "captcha_input":     "input[placeholder*='aptcha'], input[name*='captcha'], input[name*='Captcha']",
    "submit_button":     "input[type='submit'], button[type='submit']",

    # Post-login navigation — tries common link texts for attendance
    "attendance_nav": (
        "a[href*='attendance'], "
        "a:has-text('Attendance'), "
        "a:has-text('attendance'), "
        "a:has-text('ATTENDANCE')"
    ),

    # Attendance table rows
    "table_rows": "table tbody tr",
}

# Column names to look for when detecting the attendance table header
HEADER_KEYWORDS = {
    "subject":    ["subject", "course", "paper", "name"],
    "total":      ["total", "conducted", "held", "classes"],
    "attended":   ["attended", "present", "attend"],
    "percentage": ["percentage", "%", "percent"],
}


# ── Database helpers ──────────────────────────────────────────────────────────

async def clear_erp_records(user_id: str) -> int:
    """Delete all attendance records with 'ERP import' in notes for this user."""
    from app.db.session import async_session_maker
    from sqlalchemy import delete, and_
    from app.domain.models import Attendance

    async with async_session_maker() as session:
        result = await session.execute(
            delete(Attendance).where(
                and_(
                    Attendance.user_id == user_id,
                    Attendance.notes.like("ERP import%"),
                )
            )
        )
        await session.commit()
        return result.rowcount


async def store_subject_attendance(
    user_id: str,
    subject: str,
    attended: int,
    total: int,
) -> int:
    """
    Represent summary ERP data as individual present/absent records.
    All records use today's date with note 'ERP import'.
    Returns count of records stored.
    """
    from app.domain.academic import academic_repository

    today = date.today()
    absent = total - attended
    stored = 0

    for i in range(attended):
        await academic_repository.store_attendance(
            user_id=user_id,
            date=today,
            subject=subject,
            status="present",
            notes=f"ERP import ({i + 1}/{total})",
        )
        stored += 1

    for i in range(absent):
        await academic_repository.store_attendance(
            user_id=user_id,
            date=today,
            subject=subject,
            status="absent",
            notes=f"ERP import absent ({i + 1}/{absent})",
        )
        stored += 1

    return stored


# ── Playwright scraper ────────────────────────────────────────────────────────

async def scrape(
    rollno: str,
    password: str,
    headless: bool = False,
) -> List[Dict[str, Any]]:
    """
    Log in to the ERP, navigate to attendance, parse the table.
    Returns list of {subject, attended, total, percentage} dicts.
    """
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        print(
            "\n[ERROR] Playwright is not installed.\n"
            "  Run: pip install playwright && playwright install chromium\n"
        )
        sys.exit(1)

    results: List[Dict[str, Any]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        try:
            # ── Step 1: Load login page ───────────────────────────────────────
            print("  [1/4] Loading login page...")
            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60_000)

            # ── Step 2: Fill credentials ─────────────────────────────────────
            print("  [2/4] Filling credentials...")

            # Roll number — try the first visible text input
            rollno_field = page.locator(SELECTORS["rollno_input"]).first
            await rollno_field.fill(rollno)

            await page.fill(SELECTORS["password_input"], password)

            # Read and fill captcha
            captcha_text = (await page.inner_text(SELECTORS["captcha_display"])).strip()
            print(f"         Captcha detected: '{captcha_text}'")

            # Try multiple possible captcha input selectors
            captcha_filled = False
            for cap_sel in [
                "input[placeholder*='aptcha']",
                "input[name*='captcha']",
                "input[name*='Captcha']",
                "input[name*='CAPTCHA']",
                "input[id*='captcha']",
                "input[id*='Captcha']",
            ]:
                try:
                    cap_input = page.locator(cap_sel).first
                    if await cap_input.count() > 0:
                        await cap_input.fill(captcha_text)
                        captcha_filled = True
                        print(f"         Captcha input found: {cap_sel}")
                        break
                except Exception:
                    continue

            if not captcha_filled:
                # Fallback: fill the 3rd input field (after rollno + password)
                inputs = page.locator("input[type='text'], input[type='number']")
                count = await inputs.count()
                if count >= 2:
                    await inputs.nth(count - 1).fill(captcha_text)
                    print("         Captcha input: fallback to last text input")
                else:
                    print("  [WARN] Could not locate captcha input — login may fail")

            # ── Step 3: Submit and wait ──────────────────────────────────────
            print("  [3/4] Logging in...")
            await page.click(SELECTORS["submit_button"])

            # Wait for full page load including JS-rendered content
            try:
                await page.wait_for_load_state("networkidle", timeout=20_000)
            except PWTimeout:
                pass
            # Extra wait for any AJAX content after networkidle
            await asyncio.sleep(2)

            current_url = page.url
            page_text = (await page.inner_text("body")).lower()

            if "invalid" in page_text or "incorrect" in page_text or "wrong" in page_text:
                print("\n  [ERROR] Login failed — check your roll number and password.")
                await browser.close()
                return []

            if LOGIN_URL in current_url and "logout" not in page_text and "dashboard" not in page_text:
                print("\n  [ERROR] Still on login page after submit — captcha or credentials may be wrong.")
                if not headless:
                    print("          Browser is still open. Check it and press Ctrl+C to exit.")
                    await asyncio.sleep(30)
                await browser.close()
                return []

            print(f"         Logged in. URL: {current_url}")

            # ── Step 4: Extract subjects from landing page, then attendance ──
            print("  [4/4] Finding attendance data...")

            base = "https://rgipterp.com/erp/bsstums"

            # We are already on my-profile.php after login.
            # Extract subject codes from it BEFORE navigating anywhere.
            subjects = await _extract_subject_codes(page)
            print(f"         Found {len(subjects)} subjects on landing page")

            # If not found here, try loading my-profile.php explicitly
            if not subjects:
                profile_url = f"{base}/my-profile.php"
                await page.goto(profile_url, wait_until="networkidle", timeout=20_000)
                await asyncio.sleep(2)
                subjects = await _extract_subject_codes(page)
                print(f"         Found {len(subjects)} subjects on my-profile.php")

            if subjects:
                # Visit each attendance-report.php page to get present/absent counts
                for sub_code, sub_name in subjects:
                    report_url = f"{base}/attendance-report.php?sub_code={sub_code}"
                    try:
                        await page.goto(report_url, wait_until="networkidle", timeout=20_000)
                        await asyncio.sleep(1)
                        subject_results = await _parse_attendance_report_page(page, sub_name)
                        if subject_results:
                            results.extend(subject_results)
                            r = subject_results[0]
                            print(f"         {sub_name}: {r['attended']}/{r['total']} ({r['percentage']}%)")
                        else:
                            print(f"         {sub_name}: no data found on report page")
                    except Exception as e:
                        print(f"         [SKIP] {sub_code}: {e}")
                        continue
            else:
                # Last resort: rgipt_course.php also has the same subject links
                course_url = f"{base}/rgipt_course.php"
                await page.goto(course_url, wait_until="networkidle", timeout=20_000)
                await asyncio.sleep(2)
                subjects = await _extract_subject_codes(page)
                print(f"         Found {len(subjects)} subjects on rgipt_course.php")
                for sub_code, sub_name in subjects:
                    report_url = f"{base}/attendance-report.php?sub_code={sub_code}"
                    try:
                        await page.goto(report_url, wait_until="networkidle", timeout=20_000)
                        await asyncio.sleep(1)
                        subject_results = await _parse_attendance_report_page(page, sub_name)
                        if subject_results:
                            results.extend(subject_results)
                            r = subject_results[0]
                            print(f"         {sub_name}: {r['attended']}/{r['total']} ({r['percentage']}%)")
                    except Exception as e:
                        print(f"         [SKIP] {sub_code}: {e}")
                        continue
                if not results:
                    results = await _parse_attendance_table(page)

            if not results:
                print(
                    "\n  [WARN] No attendance data found.\n"
                    "         Saving page HTML for inspection..."
                )
                debug_path = Path(__file__).parent / "_erp_debug.html"
                debug_path.write_text(await page.content(), encoding="utf-8")
                print(f"         Saved to: {debug_path}")
                print("         Share this file to get help with selector updates.")

        except Exception as e:
            print(f"\n  [ERROR] Unexpected error during scrape: {e}")
            if not headless:
                print("  Browser left open for 20s so you can inspect it.")
                await asyncio.sleep(20)
        finally:
            await browser.close()

    return results


async def _parse_attendance_table(page) -> List[Dict[str, Any]]:
    """
    Find and parse an attendance table on the current page.
    Auto-detects column positions from header row.
    """
    # Find all tables on the page
    tables = page.locator("table")
    table_count = await tables.count()

    for t_idx in range(table_count):
        table = tables.nth(t_idx)
        rows = table.locator("tr")
        row_count = await rows.count()
        if row_count < 2:
            continue

        # Try to detect the header row
        header_text = (await rows.nth(0).inner_text()).lower()
        col_map = _detect_columns(header_text)

        if col_map.get("subject") is not None and col_map.get("attended") is not None:
            # Found the attendance table
            results = []
            for r_idx in range(1, row_count):
                row = rows.nth(r_idx)
                cells = row.locator("td")
                cell_count = await cells.count()
                if cell_count == 0:
                    continue

                try:
                    subject = (await cells.nth(col_map["subject"]).inner_text()).strip()
                    attended_text = (await cells.nth(col_map["attended"]).inner_text()).strip()
                    total_text = (await cells.nth(col_map["total"]).inner_text()).strip() if col_map.get("total") is not None else "0"

                    attended = _parse_int(attended_text)
                    total = _parse_int(total_text)

                    if not subject or total == 0:
                        continue

                    pct = round(attended / total * 100, 1) if total > 0 else 0.0
                    # If a percentage column exists, prefer it
                    if col_map.get("percentage") is not None:
                        pct_text = (await cells.nth(col_map["percentage"]).inner_text()).strip()
                        try:
                            pct = float(pct_text.replace("%", "").strip())
                        except Exception:
                            pass

                    results.append({
                        "subject": subject,
                        "attended": attended,
                        "total": total,
                        "percentage": pct,
                    })
                except Exception:
                    continue

            if results:
                return results

    return []


async def _extract_subject_codes(page) -> List[tuple]:
    """
    Extract (sub_code, subject_name) pairs from attendance-report.php links
    on the current page. Waits up to 8s for JavaScript to render the table.
    """
    import re
    subjects = []

    # Wait for the JS-rendered attendance links to appear
    try:
        await page.wait_for_selector(
            "a[href*='attendance-report.php']",
            state="attached",
            timeout=8_000,
        )
    except Exception:
        return []

    # Find all rows that contain an attendance-report link
    rows = page.locator("tr:has(a[href*='attendance-report.php'])")
    count = await rows.count()

    for i in range(count):
        row = rows.nth(i)
        link = row.locator("a[href*='attendance-report.php']").first
        href = (await link.get_attribute("href")) or ""
        m = re.search(r"sub_code=([\w]+)", href)
        if not m:
            continue
        sub_code = m.group(1)

        # Get subject name from column 1 (col 0 = code, col 1 = name, col 2 = link)
        cells = row.locator("td")
        cell_count = await cells.count()
        sub_name = sub_code  # fallback to code
        if cell_count >= 2:
            sub_name = (await cells.nth(1).inner_text()).strip() or sub_code

        subjects.append((sub_code, sub_name))

    return subjects


async def _parse_attendance_report_page(page, subject_name: str) -> List[Dict[str, Any]]:
    """
    Parse a single attendance-report.php?sub_code=XXXX page.

    The page shows summary stats in labeled divs:
      <label>Total Quiz</label><div class="stat-val">28</div>
      <label>Present</label><div class="stat-val">23</div>
      <label>Absent</label><div class="stat-val">5</div>
      <label>Attendance %</label>  82.14%

    We read each label/stat-val pair by name to be robust to column order.
    Falls back to counting table rows if stat-val divs are absent.
    """
    # Wait for the summary stats block to appear
    try:
        await page.wait_for_selector("div.stat-val", state="attached", timeout=8_000)
    except Exception:
        pass  # fall through to table fallback

    # ── Strategy 1: label → div.stat-val pairs ───────────────────────────────
    # Collect all parent containers that hold a <label> + a .stat-val sibling
    stat_labels = page.locator("label")
    label_count = await stat_labels.count()

    present: Optional[int] = None
    absent: Optional[int] = None
    total: Optional[int] = None
    pct_float: Optional[float] = None

    for i in range(label_count):
        label_el = stat_labels.nth(i)
        label_text = (await label_el.inner_text()).strip().lower()

        # The value lives in a sibling or adjacent div.stat-val
        # Use evaluate to grab the next-sibling's text
        val_text: str = await label_el.evaluate(
            """el => {
                // Try next sibling with class stat-val
                let sib = el.nextElementSibling;
                while (sib) {
                    if (sib.classList && sib.classList.contains('stat-val'))
                        return sib.innerText || sib.textContent || '';
                    sib = sib.nextElementSibling;
                }
                // Try parent's .stat-val child
                let parent = el.parentElement;
                if (parent) {
                    let child = parent.querySelector('.stat-val');
                    if (child) return child.innerText || child.textContent || '';
                }
                return '';
            }"""
        )
        val_text = val_text.strip()

        if "present" in label_text and "absent" not in label_text:
            try:
                present = _parse_int(val_text)
            except Exception:
                pass
        elif "absent" in label_text:
            try:
                absent = _parse_int(val_text)
            except Exception:
                pass
        elif "total" in label_text:
            try:
                total = _parse_int(val_text)
            except Exception:
                pass
        elif "attendance" in label_text and "%" in label_text:
            # Value may be inside the label element itself (as trailing text)
            # or in the next sibling
            raw = (await label_el.inner_text()).strip()
            # e.g. "Attendance %  82.14%"
            import re as _re
            m = _re.search(r"([\d.]+)\s*%", val_text + " " + raw)
            if m:
                try:
                    pct_float = float(m.group(1))
                except Exception:
                    pass

    # Also try reading Attendance % from any element whose text contains the value
    if pct_float is None:
        try:
            pct_el = page.locator("div.stat-val").last
            pct_text = (await pct_el.inner_text()).strip()
            import re as _re
            m = _re.search(r"([\d.]+)", pct_text)
            if m:
                pct_float = float(m.group(1))
        except Exception:
            pass

    if present is not None and absent is not None:
        total_calc = present + absent
        if total_calc > 0:
            pct = pct_float if pct_float is not None else round(present / total_calc * 100, 1)
            return [{"subject": subject_name, "attended": present, "total": total_calc, "percentage": pct}]

    # ── Strategy 2: positional stat-val divs (order: total, present, absent) ──
    stat_divs = page.locator("div.stat-val")
    div_count = await stat_divs.count()

    if div_count >= 3:
        try:
            # Layout from debug HTML: Total Quiz | Present | Absent | Attendance %
            total_v  = _parse_int(await stat_divs.nth(0).inner_text())
            present_v = _parse_int(await stat_divs.nth(1).inner_text())
            absent_v  = _parse_int(await stat_divs.nth(2).inner_text())
            total_calc = present_v + absent_v
            if total_calc > 0:
                pct = round(present_v / total_calc * 100, 1)
                return [{"subject": subject_name, "attended": present_v, "total": total_calc, "percentage": pct}]
        except Exception:
            pass

    # ── Strategy 3: table row counting (old fallback) ────────────────────────
    rows = page.locator("table tbody tr")
    row_count = await rows.count()
    p, a = 0, 0
    for i in range(row_count):
        text = (await rows.nth(i).inner_text()).lower()
        if "present" in text:
            p += 1
        elif "absent" in text or "miss" in text:
            a += 1
    total_calc = p + a
    if total_calc > 0:
        pct = round(p / total_calc * 100, 1)
        return [{"subject": subject_name, "attended": p, "total": total_calc, "percentage": pct}]

    return []


def _detect_columns(header_text: str) -> Dict[str, Optional[int]]:
    """
    Given a tab/whitespace-separated header row text, return a dict mapping
    {subject, total, attended, percentage} to their column indices.
    """
    cells = [c.strip() for c in header_text.replace("\t", "|").replace("  ", "|").split("|") if c.strip()]
    col_map: Dict[str, Optional[int]] = {k: None for k in HEADER_KEYWORDS}

    for idx, cell in enumerate(cells):
        cell_lower = cell.lower()
        for field, keywords in HEADER_KEYWORDS.items():
            if col_map[field] is None and any(kw in cell_lower for kw in keywords):
                col_map[field] = idx

    return col_map


def _parse_int(text: str) -> int:
    """Parse a potentially messy numeric string to int."""
    import re
    match = re.search(r"\d+", text.replace(",", ""))
    return int(match.group()) if match else 0


# ── Pretty output ─────────────────────────────────────────────────────────────

def _risk(pct: float) -> str:
    if pct < 75:
        return "HIGH RISK !"
    if pct < 85:
        return "MEDIUM"
    return "OK"


def print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("  No attendance data to display.")
        return

    print()
    print(f"  {'Subject':<35} {'Attended':>8} {'Total':>6} {'%':>7}  Risk")
    print("  " + "-" * 65)
    for r in sorted(results, key=lambda x: x["percentage"]):
        print(
            f"  {r['subject']:<35} {r['attended']:>8} {r['total']:>6} "
            f"{r['percentage']:>6.1f}%  {_risk(r['percentage'])}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def sep(char="=", w=65):
    print(char * w)


async def main() -> None:
    sep()
    print("  ERP ATTENDANCE SCRAPER — rgipterp.com")
    sep()

    parser = argparse.ArgumentParser(
        description="Scrape your RGIPT ERP attendance and store it in the agent's memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (browser visible so you can watch):
  python scripts/scrape_erp_attendance.py --user-id vansh --rollno 23IT3048 --password YOUR_PASS

  # Run headless (no browser window):
  python scripts/scrape_erp_attendance.py --user-id vansh --rollno 23IT3048 --password YOUR_PASS --headless

  # Preview data without saving:
  python scripts/scrape_erp_attendance.py --user-id vansh --rollno 23IT3048 --password YOUR_PASS --dry-run
        """,
    )
    parser.add_argument("--user-id",   required=True, help="Your agent user ID (e.g. vansh)")
    parser.add_argument("--rollno",    required=True, help="Your ERP roll number (e.g. 23IT3048)")
    parser.add_argument("--password",  required=True, help="Your ERP password")
    parser.add_argument("--headless",  action="store_true", default=False, help="Run browser invisibly")
    parser.add_argument("--keep-old",  action="store_true", default=False, help="Keep previously scraped ERP records (default: replace them)")
    parser.add_argument("--dry-run",   action="store_true", default=False, help="Print data but do not save to database")
    args = parser.parse_args()

    user_id = args.user_id.strip().lower()
    headless = args.headless

    print(f"  Roll No : {args.rollno}")
    print(f"  User ID : {user_id}")
    print(f"  Mode    : {'headless' if headless else 'visible browser'}")
    print(f"  Save    : {'DRY RUN (no save)' if args.dry_run else 'Yes'}")
    print()

    # ── Scrape ────────────────────────────────────────────────────────────────
    results = await scrape(
        rollno=args.rollno.strip(),
        password=args.password,
        headless=headless,
    )

    if not results:
        print("\n[ERROR] No attendance data was retrieved. See messages above.")
        sys.exit(1)

    print_results(results)

    if args.dry_run:
        print("  [DRY RUN] Data not saved. Remove --dry-run to save.")
        sep()
        return

    # ── Initialize memory ─────────────────────────────────────────────────────
    from app.memory.memory_manager import memory_manager
    await memory_manager.initialize()

    # ── Clear old ERP records ─────────────────────────────────────────────────
    if not args.keep_old:
        cleared = await clear_erp_records(user_id)
        if cleared:
            print(f"  Cleared {cleared} old ERP records.")

    # ── Store new records ─────────────────────────────────────────────────────
    print("  Saving to database...")
    total_stored = 0
    for r in results:
        n = await store_subject_attendance(
            user_id=user_id,
            subject=r["subject"],
            attended=r["attended"],
            total=r["total"],
        )
        total_stored += n
        print(f"    {r['subject']}: {r['attended']}/{r['total']} ({r['percentage']}%) — {n} records stored")

    sep()
    print(f"  DONE. Stored {total_stored} attendance records for {len(results)} subjects.")
    print()
    print("  Now ask your agent:")
    print('    "What is my attendance percentage?"')
    print('    "Which subjects am I at risk in?"')
    print('    "Show my attendance summary"')
    sep()


if __name__ == "__main__":
    asyncio.run(main())
