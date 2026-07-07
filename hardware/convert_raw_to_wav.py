import os
import wave
import numpy as np

# =====================================================
# PROJECT ROOT
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================================
# INPUT ROOT
# =====================================================

RAW_AUDIO_PARENT = os.path.join(
    BASE_DIR,
    "raw_data",
    "audio"
)

# =====================================================
# OUTPUT ROOT
# =====================================================

WAV_OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "processed",
    "wav_audio"
)

os.makedirs(
    WAV_OUTPUT_ROOT,
    exist_ok=True
)

# =====================================================
# AUDIO SETTINGS
# =====================================================

SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 2

MIN_DURATION_SECONDS = 1.0

# =====================================================
# RAW -> WAV
# =====================================================

def convert_raw_to_wav(raw_path, wav_path):

    with open(raw_path, "rb") as f:
        raw_data = f.read()

    if len(raw_data) == 0:
        raise ValueError(
            "Empty RAW file"
        )

    audio_data = np.frombuffer(
        raw_data,
        dtype=np.int16
    )

    if len(audio_data) == 0:
        raise ValueError(
            "No audio samples found"
        )

    duration = (
        len(audio_data)
        / SAMPLE_RATE
    )

    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            f"Audio too short ({duration:.2f}s)"
        )

    with wave.open(
        wav_path,
        "wb"
    ) as wav:

        wav.setnchannels(
            CHANNELS
        )

        wav.setsampwidth(
            SAMPLE_WIDTH
        )

        wav.setframerate(
            SAMPLE_RATE
        )

        wav.writeframes(
            audio_data.tobytes()
        )

    return duration

# =====================================================
# MAIN
# =====================================================

def main():

    print("\n===================================")
    print("TRAFFIC AI RAW TO WAV CONVERTER")
    print("===================================")

    if not os.path.exists(
        RAW_AUDIO_PARENT
    ):

        print(
            f"\nInput folder not found:\n"
            f"{RAW_AUDIO_PARENT}"
        )

        return

    raw_folders = sorted([

        folder

        for folder in os.listdir(
            RAW_AUDIO_PARENT
        )

        if os.path.isdir(
            os.path.join(
                RAW_AUDIO_PARENT,
                folder
            )
        )

    ])

    if len(raw_folders) == 0:

        print(
            "\nNo audio folders found."
        )

        return

    print(
        f"\nFound {len(raw_folders)} audio folders."
    )

    total_success = 0
    total_failed = 0
    total_skipped = 0

    total_duration = 0

    # =================================================
    # LOOP THROUGH DATASET FOLDERS
    # =================================================

    for folder in raw_folders:

        raw_folder_path = os.path.join(
            RAW_AUDIO_PARENT,
            folder
        )

        # ---------------------------------
        # BUILD OUTPUT FOLDER NAME
        # ---------------------------------

        if folder.startswith(
            "audioraw_"
        ):

            suffix = folder.replace(
                "audioraw_",
                ""
            )

            wav_folder_name = (
                f"wav_audio_{suffix}"
            )

        else:

            wav_folder_name = (
                f"wav_{folder}"
            )

        wav_folder_path = os.path.join(
            WAV_OUTPUT_ROOT,
            wav_folder_name
        )

        # =================================================
        # SKIP IF ALREADY PROCESSED
        # =================================================

        if os.path.exists(
            wav_folder_path
        ):

            print(
                f"\n[SKIPPED] {folder}"
            )

            print(
                f"Reason: {wav_folder_name} already exists"
            )

            total_skipped += 1

            continue

        # =================================================
        # CREATE OUTPUT FOLDER
        # =================================================

        os.makedirs(
            wav_folder_path,
            exist_ok=True
        )

        raw_files = sorted([

            f

            for f in os.listdir(
                raw_folder_path
            )

            if f.endswith(".raw")

        ])

        print("\n-----------------------------------")
        print(f"Processing : {folder}")
        print(f"Files      : {len(raw_files)}")
        print("-----------------------------------")

        folder_success = 0
        folder_failed = 0

        for file in raw_files:

            raw_path = os.path.join(
                raw_folder_path,
                file
            )

            wav_name = file.replace(
                ".raw",
                ".wav"
            )

            wav_path = os.path.join(
                wav_folder_path,
                wav_name
            )

            try:

                duration = convert_raw_to_wav(
                    raw_path,
                    wav_path
                )

                total_duration += duration

                folder_success += 1
                total_success += 1

                print(
                    f"[OK] {wav_name} "
                    f"({duration:.2f}s)"
                )

            except Exception as e:

                folder_failed += 1
                total_failed += 1

                print(
                    f"[FAILED] {file}"
                )

                print(
                    f"Reason: {e}"
                )

        print(
            f"\nConverted : {folder_success}"
        )

        print(
            f"Failed    : {folder_failed}"
        )

    # =================================================
    # FINAL SUMMARY
    # =================================================

    print("\n===================================")
    print("CONVERSION COMPLETE")
    print("===================================")

    print(
        f"\nTotal Converted : {total_success}"
    )

    print(
        f"Total Failed    : {total_failed}"
    )

    print(
        f"Total Skipped   : {total_skipped}"
    )

    print(
        f"Total Audio Duration : "
        f"{total_duration:.2f} sec"
    )

    print(
        f"\nOutput Location:\n"
        f"{WAV_OUTPUT_ROOT}"
    )

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()