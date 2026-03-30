# parser_ittf_h2h_to_csv.py
# Run this after you have saved all profile_*.html and h2h_*.html files

import re
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Folders where you saved the scraped pages
PROFILE_FOLDER = Path("player_profiles")
H2H_FOLDER    = Path("h2h_pages")

# Output file
OUTPUT_FILE = "ittf_h2h_data_2026.csv"   # or .xlsx if you prefer

# Current date used for age calculation (update if needed)
CURRENT_DATE = datetime(2026, 2, 2)

def parse_profile_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    row = soup.find("tr", class_=lambda c: c and "fabrik_row" in c)
    if not row:
        return None
    
    cells = row.find_all("td")
    if len(cells) < 6:
        return None
    
    # === Name & ID ===
    name_cell = cells[1]
    name_text = name_cell.get_text(strip=True)
    
    name_match = re.match(r"(.+?)\s*\(#(\d+)\)", name_text)
    if name_match:
        name = name_match.group(1).strip()
        player_id = name_match.group(2)
    else:
        name = name_text.strip()
        player_id = "UNKNOWN"
    
    print(f"Parsed name: '{name}' | ID: {player_id}")
    
    # === Profile / Assoc / Gender / Age (cell 2) ===
    assoc_cell = cells[2]
    profile_text = assoc_cell.get_text(separator="\n", strip=True)
    
    assoc = ""
    gender = "M"
    age = "?"
    birth_year = ""
    
    lines = [line.strip() for line in profile_text.split("\n") if line.strip()]
    
    for line in lines:
        if len(line) == 3 and line.isupper():  # e.g. CHN
            assoc = line
        elif line.startswith("Gender:"):
            gender_part = line.split(":", 1)[1].strip()
            gender = "M" if "Male" in gender_part else "F" if "Female" in gender_part else "M"
        elif line.startswith("Age:"):
            age_str = line.split(":", 1)[1].strip()
            if age_str.isdigit():
                age = int(age_str)
        elif line.startswith("Birth Year:"):
            birth_year = line.split(":", 1)[1].strip()
    
    # Backup age calculation
    if age == "?" and birth_year.isdigit():
        try:
            age = CURRENT_DATE.year - int(birth_year)
        except:
            pass
    
    # === Career stats (cell 3) ===
    career_cell = cells[3]
    career_text = career_cell.get_text(separator="\n", strip=True)
    
    lines = [line.strip() for line in career_text.split("\n") if line.strip()]
    
    career = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line:
            key = line.split(":", 1)[0].strip()
            # Value is usually on next line
            val_line = lines[i+1] if i+1 < len(lines) else ""
            val = val_line.strip()
            try:
                career[key] = int(val.split()[0]) if val.split() else 0
            except ValueError:
                career[key] = val
            i += 2
        else:
            i += 1
    
    stats = {
        "player_id": player_id,
        "name": name,
        "assoc": assoc,
        "gender": gender,
        "age": age,
        "events": career.get("Events", 0),
        "matches": career.get("Matches", 0),
        "wins": career.get("Wins", 0),
        "losses": career.get("Loses", 0),
        "titles": career.get("WTT Senior Titles", career.get("All Senior Titles", 0)),
    }
    
    print(f"Extracted: gender={gender}, age={age}, events={stats['events']}, matches={stats['matches']}")
    
    return stats

def parse_h2h_html(html_content, player_a_name, player_b_name):
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", id="list_26_com_fabrik_26")
    if not table:
        return []
    
    matches = []
    rows = table.select("tbody tr.fabrik_row")
    
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue
        
        year     = cells[0].get_text(strip=True)          # e.g. "2024"
        event    = cells[1].get_text(strip=True)
        stage    = cells[2].get_text(strip=True)
        subevent = cells[3].get_text(strip=True)
        round_   = cells[4].get_text(strip=True)
        p1_cell  = cells[5]
        p2_cell  = cells[6]
        games    = cells[7].get_text(strip=True)
        result   = cells[8].get_text(strip=True)
        winner   = cells[9].get_text(strip=True)
        
        game_list = re.findall(r"\d+:\d+", games)
        game_list = [g.replace(":", "-") for g in game_list]
        
        if " - " in result:
            a_sets, b_sets = map(int, result.split(" - "))
        else:
            a_sets = b_sets = 0
        
        winner_id = "A" if winner.strip() == player_a_name else "B" if winner.strip() == player_b_name else "?"
        
        played_games = len(game_list)
        best_of = "Best of 3" if played_games <= 3 else "Best of 5"
        
        match_scores = game_list + ["N/A"] * (5 - len(game_list))
        
        matches.append({
            "year": year,
            "event": event,
            "stage": stage,
            "subevent": subevent,
            "round": round_,
            "games": games,
            "result": result,
            "winner": winner,
            "winner_id": winner_id,
            "best_of": best_of,
            "match_scores": match_scores,
            "sets_a": a_sets,
            "sets_b": b_sets
        })
    
    return matches

def main():
    # 1. Load all player profiles into a dict {name: stats}
    player_data = {}
    for file in PROFILE_FOLDER.glob("profile_*.html"):
        content = file.read_text(encoding="utf-8")
        stats = parse_profile_html(content)
        if stats and stats["name"]:
            player_data[stats["name"]] = stats
            print(f"Loaded profile: {stats['name']} ({stats['player_id']})")

    # 2. Prepare rows for CSV
    all_rows = []

    for file in H2H_FOLDER.glob("h2h_*.html"):
        content = file.read_text(encoding="utf-8")

        # Extract names from filename
        # Example: h2h_01_WANG_Chuqin_vs_FAN_Zhendong.html
        m = re.match(r"h2h_\d+_(.*)_vs_(.*)\.html", file.name)
        if not m:
            print(f"Skipping file with unknown name format: {file.name}")
            continue

        name_a_raw = m.group(1).replace("_", " ")
        name_b_raw = m.group(2).replace("_", " ")

        # Try to match with loaded profiles (may need fuzzy matching in real case)
        pa = player_data.get(name_a_raw)
        pb = player_data.get(name_b_raw)

        if not pa or not pb:
            print(f"Missing profile data for {name_a_raw} or {name_b_raw} → skipping")
            continue

        print(f"Found profile data for {name_a_raw} or {name_b_raw} → skipping")

        # Parse matches
        matches = parse_h2h_html(content, name_a_raw, name_b_raw)

        for match in matches:
            row = {
                "Year": match["year"],
                "Match Score": match["result"],
                "Player A": pa["name"],
                "ID_A": pa["player_id"],
                "Gender_A": pa["gender"],
                "Age_A": pa["age"],
                "Events_A": pa["events"],
                "Matches_A": pa["matches"],
                "Wins_A": pa["wins"],
                "Losses_A": pa["losses"],
                "Titles_A": pa["titles"],

                "Player B": pb["name"],
                "ID_B": pb["player_id"],
                "Gender_B": pb["gender"],
                "Age_B": pb["age"],
                "Events_B": pb["events"],
                "Matches_B": pb["matches"],
                "Wins_B": pb["wins"],          # note: column name has typo in your template
                "Losses_B": pb["losses"],
                "Titles_B": pb["titles"],

                "Winner": match["winner"],
                "Score": f'"{match["result"]}"',                 # ← add quotes
                "Head-To-Head": f'"{match["sets_a"]}-{match["sets_b"]}"',
                "Best of 3/5": match["best_of"],
                "Match_score1": f'"{match["match_scores"][0]}"' if match["match_scores"][0] != "N/A" else "N/A",
                "Match_score2": f'"{match["match_scores"][1]}"' if match["match_scores"][1] != "N/A" else "N/A",
                "Match_score3": f'"{match["match_scores"][2]}"' if match["match_scores"][2] != "N/A" else "N/A",
                "Match_score4": f'"{match["match_scores"][3]}"' if match["match_scores"][3] != "N/A" else "N/A",
                "Match_score5": f'"{match["match_scores"][4]}"' if match["match_scores"][4] != "N/A" else "N/A",
            }
            all_rows.append(row)

    if not all_rows:
        print("No matches parsed. Check file names / content.")
        return

    # 3. Create DataFrame & save
    df = pd.DataFrame(all_rows)
    # Reorder columns exactly as in your template
    desired_columns = [
        "Year",
        "Match Score", "Player A", "ID_A", "Gender_A", "Age_A", "Events_A", "Matches_A", "Wins_A", "Losses_A", "Titles_A",
        "Player B", "ID_B", "Gender_B", "Age_B", "Events_B", "Matches_B", "Wins_B", "Losses_B", "Titles_B",
        "Winner", "Score", "Head-To-Head", "Best of 3/5",
        "Match_score1", "Match_score2", "Match_score2", "Match_score3", "Match_score4", "Match_score5"
    ]
    # Keep only columns that exist, fill missing with NaN
    df = df.reindex(columns=desired_columns)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")   # make sure it's numeric
    df = df.sort_values(by="Year", ascending=True).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")

    # Optional: also save as Excel
    df.to_excel(OUTPUT_FILE.replace(".csv", ".xlsx"), index=False)
    print(f"Also saved Excel version")



if __name__ == "__main__":
    main()