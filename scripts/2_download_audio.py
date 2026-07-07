
import pandas as pd
import subprocess
import os

# =====================================================
# CONFIGURATION
# =====================================================
ID_TO_NAME = {
    '/m/0k4j': 'car',
    '/m/01bjv': 'bus',
    '/m/04_sv': 'bike',
    '/m/07r04': 'truck'
}

BATCH_SIZE = 500 # For remaining categories
MAX_FILES_PER_CATEGORY = 10270 # Cap each category at this many files

new_downloads = {'car': 0, 'bus': 0, 'bike': 0, 'truck': 0, 'mixed_vehicles': 0}
historical_skips = {'car': 0, 'bus': 0, 'bike': 0, 'truck': 0, 'mixed_vehicles': 0}

# Paths
input_csv = os.path.join('processed', 'filtered_vehicle_segments.csv')
BASE_DIR = os.path.join("processed", "wav_audio")
DATASET_CSV = os.path.join("processed", "datasets", "final_audio_dataset.csv")
FAILED_LOG_FILE = os.path.join(BASE_DIR, "failed_ytids.txt")

os.makedirs(BASE_DIR, exist_ok=True)

print("=" * 50)
print("SMART AUDIO DOWNLOADER (PRIORITIZED)")
print("=" * 50)

# =====================================================
# 1. LOAD BLACKLIST (Dead Links)
# =====================================================
failed_ytids = set()
if os.path.exists(FAILED_LOG_FILE):
    with open(FAILED_LOG_FILE, 'r') as f:
        failed_ytids = set(line.strip() for line in f)
    print(f"[MEMORY] Loaded {len(failed_ytids)} dead YouTube IDs to skip.")

# =====================================================
# 2. LOAD EXTRACTED DATASET (To skip re-downloading)
# =====================================================
extracted_files = set()
if os.path.exists(DATASET_CSV):
    try:
        feat_df = pd.read_csv(DATASET_CSV, usecols=['file_name'])
        extracted_files = set(feat_df['file_name'].tolist())
        print(f"[MEMORY] Loaded {len(extracted_files)} already-extracted files.")
    except Exception as e:
        print(f"Warning: Could not load feature dataset: {e}")

# =====================================================
# 3. PREPARE & CATEGORIZE DATA
# =====================================================
try:
    df = pd.read_csv(input_csv)
    print(f"Goal: Download ALL {len(df)} files...\n")
    
    # Shuffle the dataframe initially
    df = df.sample(frac=1).reset_index(drop=True)
except FileNotFoundError:
    print(f"ERROR: Could not find {input_csv}.")
    exit()

# Helper function to get current count of files on disk for a category
def get_current_count(cat):
    cat_dir = os.path.join(BASE_DIR, cat)
    if not os.path.exists(cat_dir):
        return 0
    return len([f for f in os.listdir(cat_dir) if f.endswith('.wav')])

# Helper function to categorize a row
def get_category(labels):
    labels_str = str(labels)
    for ont_id, name in ID_TO_NAME.items():
        if ont_id in labels_str:
            return name
    return "mixed_vehicles"

# Assign categories and split dataframes
df['assigned_category'] = df['positive_labels'].apply(get_category)
category_dfs = {cat: group.reset_index(drop=True) for cat, group in df.groupby('assigned_category')}
category_pointers = {cat: 0 for cat in category_dfs.keys()}

active_categories = list(category_dfs.keys())

# =====================================================
# 4. DOWNLOAD FUNCTION
# =====================================================
def download_single_file(cat, row):
    ytid = row['YTID']
    start = row['start_seconds']
    end = row['end_seconds']
    
    if ytid in failed_ytids:
        return False
    
    save_dir = os.path.join(BASE_DIR, cat)
    os.makedirs(save_dir, exist_ok=True)
    
    target_filename = f"{ytid}_{start}_{end}.wav"
    output_filepath = os.path.join(save_dir, target_filename)
    
    if target_filename in extracted_files or os.path.exists(output_filepath):
        historical_skips[cat] += 1
        return False
    
    youtube_url = f"https://www.youtube.com/watch?v={ytid}"
    total_saved = historical_skips[cat] + new_downloads[cat] + 1
    total_cat_files = min(len(category_dfs[cat]), MAX_FILES_PER_CATEGORY)
    print(f"Downloading {cat.upper()} [Total on disk: {total_saved}/{total_cat_files}]: {ytid}...")
    
    command = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", output_filepath,
        youtube_url
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            new_downloads[cat] += 1
            return True
        else:
            print(f"  -> Skipped (Video unavailable/private). Blacklisting ID...")
            failed_ytids.add(ytid)
            with open(FAILED_LOG_FILE, 'a') as f:
                f.write(f"{ytid}\n")
            return False
    except Exception as e:
        print(f"  -> Failed to download {ytid}: {e}")
        return False

# =====================================================
# 5. PHASE 1: DOWNLOAD BIKE AND TRUCK FULLY FIRST
# =====================================================
priority_categories = ['bike', 'truck']
for cat in priority_categories:
    if cat not in active_categories:
        continue
        
    df_cat = category_dfs[cat]
    print(f"\n{'='*50}")
    print(f"PHASE 1: DOWNLOADING {cat.upper()} FULLY")
    print(f"{'='*50}")
    
    while True:
        # Check max cap
        current_count = get_current_count(cat)
        if current_count >= MAX_FILES_PER_CATEGORY:
            print(f"\n✅ CATEGORY CAP REACHED: {cat.upper()} has {current_count}/{MAX_FILES_PER_CATEGORY} files.")
            active_categories.remove(cat)
            break
            
        # Check if no more rows
        if category_pointers[cat] >= len(df_cat):
            print(f"\n✅ Finished ALL available files for category: {cat.upper()}")
            active_categories.remove(cat)
            break
            
        row_idx = category_pointers[cat]
        row = df_cat.iloc[row_idx]
        category_pointers[cat] += 1
        
        download_single_file(cat, row)

# =====================================================
# 6. PHASE 2: DOWNLOAD REMAINING CATEGORIES
# =====================================================
remaining_categories = [cat for cat in active_categories if cat not in priority_categories]
if remaining_categories:
    print(f"\n{'='*50}")
    print(f"PHASE 2: DOWNLOADING REMAINING CATEGORIES")
    print(f"{'='*50}")
    round_num = 1
    
    while active_categories:
        print(f"\n--- ROUND {round_num} ---")
        for cat in list(remaining_categories):
            if cat not in active_categories:
                continue
                
            df_cat = category_dfs[cat]
            print(f"\nProcessing {cat.upper()}...")
            
            processed_in_batch = 0
            
            while processed_in_batch < BATCH_SIZE and category_pointers[cat] < len(df_cat):
                # Check max cap
                current_count = get_current_count(cat)
                if current_count >= MAX_FILES_PER_CATEGORY:
                    print(f"\n✅ CATEGORY CAP REACHED: {cat.upper()} has {current_count}/{MAX_FILES_PER_CATEGORY} files.")
                    active_categories.remove(cat)
                    break
                    
                row_idx = category_pointers[cat]
                row = df_cat.iloc[row_idx]
                category_pointers[cat] += 1
                
                downloaded = download_single_file(cat, row)
                if downloaded:
                    processed_in_batch += 1
                    
            if category_pointers[cat] >= len(df_cat):
                print(f"✅ Finished ALL available files for category: {cat.upper()}")
                active_categories.remove(cat)
                
        round_num += 1

# =====================================================
# 7. SESSION SUMMARY
# =====================================================
print("\n" + "=" * 50)
print("SESSION SUMMARY")
print("=" * 50)
for cat in new_downloads.keys():
    if cat in category_dfs or cat == 'mixed_vehicles':
        print(f"{cat.capitalize()}: {new_downloads.get(cat, 0)} New Downloads (Bypassed {historical_skips.get(cat, 0)} historical files)")
