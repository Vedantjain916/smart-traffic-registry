import os
import numpy as np
import pandas as pd

# =====================================================
# PROJECT ROOT
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================================
# INPUT / OUTPUT
# =====================================================

VIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "raw_data",
    "vibration"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "processed",
    "vibration_features"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def calculate_features(group):

    magnitude = (
        group["magnitude"]
        .astype(float)
        .values
    )

    vibration_signal = np.abs(
        magnitude - 9.81
    )

    if len(vibration_signal) == 0:

        raise ValueError(
            "No vibration samples found"
        )

    vib_mean = np.mean(
        vibration_signal
    )

    vib_std = np.std(
        vibration_signal
    )

    vib_rms = np.sqrt(
        np.mean(
            vibration_signal ** 2
        )
    )

    vib_max = np.max(
        vibration_signal
    )

    vib_min = np.min(
        vibration_signal
    )

    vib_peak_to_peak = (
        vib_max -
        vib_min
    )

    return (

        vib_mean,

        vib_std,

        vib_rms,

        vib_max,

        vib_min,

        vib_peak_to_peak

    )

# =====================================================
# PROCESS SINGLE FILE
# =====================================================

def process_vibration_file(csv_path):

    print(
        f"\nReading: {csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    required_columns = [

        "magnitude",

        "audio_file"

    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing column: {col}"
            )

    grouped = df.groupby(
        "audio_file"
    )

    rows = []

    for audio_file, group in grouped:

        (

            vib_mean,

            vib_std,

            vib_rms,

            vib_max,

            vib_min,

            vib_peak_to_peak

        ) = calculate_features(
            group
        )

        rows.append({

            "audio_file":
                os.path.basename(
                    audio_file
                ).replace(
                    ".raw",
                    ".wav"
                ),

            "vib_mean":
                float(vib_mean),

            "vib_std":
                float(vib_std),

            "vib_rms":
                float(vib_rms),

            "vib_max":
                float(vib_max),

            "vib_min":
                float(vib_min),

            "vib_peak_to_peak":
                float(vib_peak_to_peak)

        })

    return pd.DataFrame(
        rows
    )

# =====================================================
# MAIN
# =====================================================

def main():

    print("\n====================================")
    print("VIBRATION FEATURE EXTRACTION")
    print("====================================")

    if not os.path.exists(
        VIBRATION_FOLDER
    ):

        print(
            f"\nInput folder not found:\n{VIBRATION_FOLDER}"
        )

        return

    vibration_files = sorted([

        f

        for f in os.listdir(
            VIBRATION_FOLDER
        )

        if f.endswith(".csv")

    ])

    if len(vibration_files) == 0:

        print(
            "\nNo vibration CSV files found."
        )

        return

    total_records = 0
    total_skipped = 0

    for file in vibration_files:

        try:

            suffix = file.replace(
                "vibration_",
                ""
            ).replace(
                ".csv",
                ""
            )

            output_file = os.path.join(

                OUTPUT_FOLDER,

                f"vibration_features_{suffix}.csv"

            )

            # =====================================
            # SKIP IF ALREADY PROCESSED
            # =====================================

            if os.path.exists(
                output_file
            ):

                print(
                    f"\n[SKIPPED] {file}"
                )

                print(
                    f"Reason: vibration_features_{suffix}.csv already exists"
                )

                total_skipped += 1

                continue

            file_path = os.path.join(
                VIBRATION_FOLDER,
                file
            )

            features_df = process_vibration_file(
                file_path
            )

            features_df.to_csv(

                output_file,

                index=False

            )

            total_records += len(
                features_df
            )

            print(
                f"\nSaved: {output_file}"
            )

            print(
                f"Rows: {len(features_df)}"
            )

        except Exception as e:

            print(
                f"\nFailed: {file}"
            )

            print(e)

    print("\n====================================")
    print("COMPLETE")
    print("====================================")

    print(
        f"Total Event Records: {total_records}"
    )

    print(
        f"Files Skipped: {total_skipped}"
    )

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()