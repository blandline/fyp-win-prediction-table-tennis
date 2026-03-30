import itertools
import os
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ========== CONFIG ==========

USERNAME = "hhkhanhh1256"
PASSWORD = "Ahmad@442177" 
# authentication details excluded here

LOGIN_URL = "https://results.ittf.link/index.php/login"
RANKINGS_URL = (
    "https://www.results.ittf.link/index.php/ittf-rankings/ittf-ranking-men-singles/list/57?resetfilters=0&clearordering=0&clearfilters=0"
)

# Direct Fabrik list endpoint for Head-to-Head
H2H_LIST_URL = (
    "https://results.ittf.link/index.php/head-to-head/list/26"
    "?resetfilters=0&clearordering=0&clearfilters=0"
)

OUTPUT_DIR = Path("h2h_pages")
OUTPUT_DIR.mkdir(exist_ok=True)

# Table column indices on rankings page
# # | pos | diff | points | Name | Flag | Assoc | Continent | ...
RANK_NAME_COL = 4
RANK_ASSOC_COL = 6

DEFAULT_WAIT = 15


# ========== DRIVER ==========

def make_driver(headless: bool = False):
    options = Options()
    if headless:
        options.add_argument("-headless")
    service = Service(
        r"C:\Users\msarbulan2\Documents\scrapping\geckodriver.exe"
    )
    driver = webdriver.Firefox(service=service, options=options)
    driver.implicitly_wait(5)
    return driver


# ========== HELPERS ==========

def accept_cookies_if_any(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//*[self::button or self::a][contains(., 'I agree')]")
            )
        ).click()
    except Exception:
        pass


def login(driver):
    driver.get(LOGIN_URL)
    accept_cookies_if_any(driver)

    wait = WebDriverWait(driver, DEFAULT_WAIT)
    user_input = wait.until(EC.presence_of_element_located((By.NAME, "username")))
    pwd_input = driver.find_element(By.NAME, "password")

    user_input.clear()
    user_input.send_keys(USERNAME)
    pwd_input.clear()
    pwd_input.send_keys(PASSWORD)

    login_btn = driver.find_element(
        By.XPATH, "//button[@type='submit' or contains(., 'Log in')]"
    )
    login_btn.click()

    wait.until(lambda d: "login" not in d.current_url.lower())


def get_top_n_players_with_ids(driver, start_rank=1, num_players=100):
    driver.get(RANKINGS_URL)
    accept_cookies_if_any(driver)

    wait = WebDriverWait(driver, DEFAULT_WAIT)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))

    # Try to set records per page to 100
    limit_set = False
    for limit_id in ['limit57', 'list_57_com_fabrik_57___limit', 'fabrik___limit', 'limit']:
        try:
            limit_select = driver.find_element(By.ID, limit_id)
            limit_select.send_keys("100")
            driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", limit_select)
            time.sleep(2)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
            print(f"Page size set to 100 using ID: {limit_id}")
            limit_set = True
            break
        except:
            continue

    if not limit_set:
        print("Could not change page size → using default (~50 rows/page)")

    page_size = 100 if limit_set else 50

    # ──────────────────────────────────────────────
    # Define selectors HERE (outside any loop)
    selectors = [
        '.fabrikNav .pagination-next a:not(.disabled)',
        'ul.pagination li.next a',
        'ul.pagination li.page-item.next a',
        'a[rel="next"]',
        '.pagination .next a',
        'a[title="Next"]',
        '.fabrik_list-footer a.next'
    ]
    # ──────────────────────────────────────────────

    # Calculate how many full pages to skip
    pages_to_skip = (start_rank - 1) // page_size

    # Skip to starting page
    for i in range(pages_to_skip):
        print(f"Navigating to page {i+2}...")
        next_clicked = False
        for sel in selectors:
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
                print(f"  → Found Next using selector: {sel}")
                driver.execute_script("arguments[0].click();", next_btn)
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 0)
                time.sleep(1.5)
                next_clicked = True
                break
            except TimeoutException:
                continue

        if not next_clicked:
            print("Could not find or click Next button → stopping early")
            break

    # Now collect players
    players = []
    collected = 0
    skip_on_first = (start_rank - 1) % page_size

    while collected < num_players:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        print(f"  Page has {len(rows)} rows")

        start_idx = skip_on_first if collected == 0 else 0
        for row in rows[start_idx:]:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) <= max(RANK_NAME_COL, RANK_ASSOC_COL):
                continue

            name_cell = cells[RANK_NAME_COL]
            try:
                link = name_cell.find_element(By.TAG_NAME, "a")
            except:
                continue

            name = link.text.strip()
            href = link.get_attribute("href") or ""
            assoc = cells[RANK_ASSOC_COL].text.strip()

            m = re.search(r"player_id_raw=(\d+)", href)
            if not m:
                continue
            player_id = m.group(1)

            display = f"{name} ({assoc})" if assoc else name

            players.append({
                "name": name,
                "assoc": assoc,
                "display": display,
                "player_id": player_id,
            })
            collected += 1
            if collected >= num_players:
                break

        if collected >= num_players:
            break

        # Try to go to next page
        next_clicked = False
        for sel in selectors:
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
                print(f"  → Clicking Next using: {sel}")
                driver.execute_script("arguments[0].click();", next_btn)
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 0)
                time.sleep(1.5)
                next_clicked = True
                break
            except TimeoutException:
                continue

        if not next_clicked:
            print("No more pages or Next button not found")
            break

    print(f"Collected {len(players)} players starting from rank {start_rank}")
    return players


def sanitize_filename(text: str) -> str:
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^\w_.-]", "", text)


def fetch_h2h_html_for_pair(driver, pa, pb, pair_index, total_pairs):
    """
    - Open H2H list page
    - Inject numeric IDs and text via JS into hidden/text inputs
    - Submit form
    - Wait for results table
    - Return page_source
    """
    pa_id = pa["player_id"]
    pb_id = pb["player_id"]
    pa_disp = pa["display"]
    pb_disp = pb["display"]

    print(f"[{pair_index}/{total_pairs}] {pa_disp} vs {pb_disp}")

    driver.get(H2H_LIST_URL)
    accept_cookies_if_any(driver)

    # Ensure form is present
    WebDriverWait(driver, DEFAULT_WAIT).until(
        EC.presence_of_element_located((By.ID, "listform_26_com_fabrik_26"))
    )

    # Prepare the display text exactly as the site expects, e.g. "WANG Chuqin (CHN)"
    pa_text = pa_disp
    pb_text = pb_disp

    # Inject values into hidden numeric fields + visible autocomplete text inputs
    js = """
    const form = document.getElementById('listform_26_com_fabrik_26');
    if (!form) return 'NO_FORM';

    function setVal(sel, val) {
      const el = form.querySelector(sel);
      if (el) { el.value = val; }
    }

    // numeric IDs for Player A (index 0) and Player B (index 1)
    setVal("input[name='fabrik___filter[list_26_com_fabrik_26][value][0]']", arguments[0]);
    setVal("input[name='fabrik___filter[list_26_com_fabrik_26][value][1]']", arguments[1]);

    // visible text inputs (autocomplete)
    setVal("input[name='auto-complete234']", arguments[2]);
    setVal("input[name='auto-complete236']", arguments[3]);

    form.submit();
    return 'OK';
    """

    res = driver.execute_script(js, pa_id, pb_id, pa_text, pb_text)
    if res != "OK":
        print(f"  JS injection returned: {res}")
        return None

    # Wait for the result table to appear (or show "No records")
    try:
        WebDriverWait(driver, DEFAULT_WAIT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table#list_26_com_fabrik_26")
            )
        )
    except TimeoutException:
        print("  H2H result table did not appear (timeout); saving page anyway.")

    time.sleep(1.0)  # small extra settle time
    return driver.page_source

def fetch_h2h_html_for_pair(driver, pa, pb, pair_index, total_pairs):
    """
    - Open H2H list page
    - Inject numeric IDs and text via JS into hidden/text inputs
    - Submit form
    - Wait for results table
    - Return page_source
    """
    pa_id = pa["player_id"]
    pb_id = pb["player_id"]
    pa_disp = pa["display"]
    pb_disp = pb["display"]

    print(f"[{pair_index}/{total_pairs}] {pa_disp} vs {pb_disp}")

    driver.get(H2H_LIST_URL)
    accept_cookies_if_any(driver)

    # Ensure form is present
    WebDriverWait(driver, DEFAULT_WAIT).until(
        EC.presence_of_element_located((By.ID, "listform_26_com_fabrik_26"))
    )

    # Prepare the display text exactly as the site expects, e.g. "WANG Chuqin (CHN)"
    pa_text = pa_disp
    pb_text = pb_disp

    # Inject values into hidden numeric fields + visible autocomplete text inputs
    js = """
    const form = document.getElementById('listform_26_com_fabrik_26');
    if (!form) return 'NO_FORM';

    function setVal(sel, val) {
      const el = form.querySelector(sel);
      if (el) { el.value = val; }
    }

    // numeric IDs for Player A (index 0) and Player B (index 1)
    setVal("input[name='fabrik___filter[list_26_com_fabrik_26][value][0]']", arguments[0]);
    setVal("input[name='fabrik___filter[list_26_com_fabrik_26][value][1]']", arguments[1]);

    // visible text inputs (autocomplete)
    setVal("input[name='auto-complete234']", arguments[2]);
    setVal("input[name='auto-complete236']", arguments[3]);

    form.submit();
    return 'OK';
    """

    res = driver.execute_script(js, pa_id, pb_id, pa_text, pb_text)
    if res != "OK":
        print(f"  JS injection returned: {res}")
        return None

    # Wait for the result table to appear (or show "No records")
    try:
        WebDriverWait(driver, DEFAULT_WAIT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table#list_26_com_fabrik_26")
            )
        )
    except TimeoutException:
        print("  H2H result table did not appear (timeout); saving page anyway.")

    time.sleep(1.0)  # small extra settle time
    return driver.page_source

def fetch_player_profile_html(driver, player, index, total):
    """
    Opens the players-profiles list (listid=33), injects the player_id into the filter,
    submits the form, waits for the result, and returns the page source.

    Args:
        driver: selenium WebDriver
        player: dict with at least 'player_id' and 'display' keys (same as you use)
        index: current number (for logging)
        total: total count (for logging)

    Returns:
        str | None : HTML page source or None if serious failure
    """
    player_id = player["player_id"]
    display = player["display"]  # e.g., "LIN Shidong (CHN)"

    print(f"[{index}/{total}] Fetching profile: {display} (id={player_id})")

    PROFILE_LIST_URL = (
        "https://results.ittf.link/index.php/players-profiles/list/33"
        "?resetfilters=0&clearordering=0&clearfilters=0"
    )

    driver.get(PROFILE_LIST_URL)
    accept_cookies_if_any(driver)

    try:
        # Wait for the filter form to be present
        WebDriverWait(driver, DEFAULT_WAIT).until(
            EC.presence_of_element_located((By.ID, "listform_33_com_fabrik_33"))
        )
    except TimeoutException:
        print("  Player profile form not found → possibly wrong list id or layout change")
        return None

    # JavaScript injection with exact names from the HTML
    js = """
    const form = document.getElementById('listform_33_com_fabrik_33');
    if (!form) return 'NO_FORM';

    function setVal(sel, val) {
      const el = form.querySelector(sel);
      if (el) { el.value = val; }
    }

    // Hidden numeric ID filter
    setVal("input[name='fabrik___filter[list_33_com_fabrik_33][value][0]']", arguments[0]);

    // Visible autocomplete text field
    setVal("input[name='auto-complete441']", arguments[1]);

    form.submit();
    return 'OK';
    """

    res = driver.execute_script(js, player_id, display)
    if res != "OK":
        print(f"  JS injection returned: {res}")
        return None

    # Wait for the filtered table to have actual data rows (not the "please select" message)
    try:
        WebDriverWait(driver, DEFAULT_WAIT).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, "table#list_33_com_fabrik_33 tbody tr.fabrik_row")) > 0
            or "No records found" in d.page_source
        )
        print("  Profile table loaded with data")
    except TimeoutException:
        print("  Timeout waiting for filtered results → saving anyway")

    time.sleep(2.0)  # extra settle time for any JS to finish

    return driver.page_source



def main():
    driver = make_driver(headless=False)
    try:
        print("Logging in...")
        login(driver)

        print("Collecting players from rank 38 to 100...")
        # 100 - 38 + 1 = 63
        players = get_top_n_players_with_ids(driver, start_rank=42, num_players=59)
        for p in players:
            print(f"  - {p['display']} (id={p['player_id']})")

        pairs = list(itertools.combinations(players, 2))
        total_pairs = len(pairs)
        print(f"Total pairs: {total_pairs}")

        for idx, (pa, pb) in enumerate(pairs, start=1):
            try:
                html = fetch_h2h_html_for_pair(driver, pa, pb, idx, total_pairs)
                if not html:
                    continue

                fname = (
                    f"h2h_{idx:02d}_"
                    f"{sanitize_filename(pa['name'])}_vs_"
                    f"{sanitize_filename(pb['name'])}.html"
                )
                out_path = OUTPUT_DIR / fname
                out_path.write_text(html, encoding="utf-8")
                print(f"  Saved {out_path}")
            except Exception as e:
                print(f"  Error for {pa['display']} vs {pb['display']}: {e}")
                continue

        print("Done. HTML files in:", OUTPUT_DIR.resolve())
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# def main():
#     driver = make_driver(headless=False)
#     try:
#         print("Logging in...")
#         login(driver)

#         print("Collecting top players...")
#         players = get_top_n_players_with_ids(driver, start_rank=56, num_players=45)
#         for p in players:
#             print(f"  - {p['display']} (id={p['player_id']})")

#         total = len(players)

#         OUTPUT_DIR = Path("player_profiles")
#         OUTPUT_DIR.mkdir(exist_ok=True)

#         for idx, player in enumerate(players, start=1):
#             try:
#                 html = fetch_player_profile_html(driver, player, idx, total)
#                 if not html:
#                     continue

#                 fname = f"profile_{idx:03d}_{sanitize_filename(player['name'])}.html"
#                 out_path = OUTPUT_DIR / fname
#                 out_path.write_text(html, encoding="utf-8")
#                 print(f"  Saved → {out_path}")

#             except Exception as e:
#                 print(f"  Error for {player['display']}: {e}")
#                 continue

#         print("Done. Profiles saved in:", OUTPUT_DIR.resolve())

#     finally:
#         driver.quit()

if __name__ == "__main__":
    main()
