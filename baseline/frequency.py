# =========================
# Example parsed Top-5 results
# =========================
# =========================
# Top-5 results (mean ± std already done)
# =========================
Top5 = {
    "M3CV": {
        "ME": {
            "band": ["30-32", "12-14", "28-30", "10-12", "24-26"],
            "chan": ["Ch31", "Ch56", "Ch30", "Ch57", "Ch2"],
        },
        "RS": {
            "band": ["10-12", "12-14", "30-32", "8-10", "28-30"],
            "chan": ["Ch56", "Ch30", "Ch31", "Ch57", "Ch29"],
        },
        "SSS": {
            "band": ["30-32", "28-30", "10-12", "20-22", "12-14"],
            "chan": ["Ch57", "Ch56", "Ch30", "Ch31", "Ch28"],
        },
        "TSS": {
            "band": ["30-32", "10-12", "26-28", "8-10", "24-26"],
            "chan": ["Ch30", "Ch31", "Ch56", "Ch28", "Ch57"],
        },
        "P300": {
            "band": ["10-12", "12-14", "20-22", "24-26", "28-30"],
            "chan": ["Ch31", "Ch30", "Ch56", "Ch57", "Ch2"],
        },
        "SSA": {
            "band": ["10-12", "30-32", "28-30", "12-14", "26-28"],
            "chan": ["Ch31", "Ch56", "Ch30", "Ch57", "Ch29"],
        },
    },

    "OpenBMI": {
        "MI": {
            "band": ["10-12", "12-14", "8-10", "14-16", "18-20"],
            "chan": ["ChC3", "ChC4", "ChCz", "ChCP3", "ChCP4"],
        },
        "SSVEP": {
            "band": ["8-10", "10-12", "12-14", "14-16", "16-18"],
            "chan": ["ChO1", "ChO2", "ChOz", "ChPOz", "ChPO3"],
        },
    }
}



from collections import Counter
import pandas as pd
import os

SAVE_DIR = "./quantified_contribution"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Frequency counting
# =========================
def count_global_frequency(top5):
    band_counter = Counter()
    chan_counter = Counter()

    for dataset in top5:
        for task in top5[dataset]:
            band_counter.update(top5[dataset][task]["band"])
            chan_counter.update(top5[dataset][task]["chan"])

    return band_counter, chan_counter


def count_dataset_frequency(top5):
    result = {}

    for dataset in top5:
        band_c = Counter()
        chan_c = Counter()
        for task in top5[dataset]:
            band_c.update(top5[dataset][task]["band"])
            chan_c.update(top5[dataset][task]["chan"])
        result[dataset] = {"band": band_c, "chan": chan_c}

    return result


# =========================
# Convert Counter → DataFrame
# =========================
def counter_to_df(counter, name):
    df = pd.DataFrame(
        counter.most_common(),
        columns=[name, "Frequency"]
    )
    return df


# =========================
# Run statistics
# =========================
band_all, chan_all = count_global_frequency(Top5)
by_dataset = count_dataset_frequency(Top5)

# =========================
# Save global results
# =========================
df_band_all = counter_to_df(band_all, "Subband(Hz)")
df_chan_all = counter_to_df(chan_all, "Channel")

df_band_all.to_csv(
    os.path.join(SAVE_DIR, "Global_Subband_Frequency.csv"),
    index=False
)
df_chan_all.to_csv(
    os.path.join(SAVE_DIR, "Global_Channel_Frequency.csv"),
    index=False
)

print("Saved global frequency tables.")

# =========================
# Save per-dataset results
# =========================
for dataset in by_dataset:
    df_band = counter_to_df(by_dataset[dataset]["band"], "Subband(Hz)")
    df_chan = counter_to_df(by_dataset[dataset]["chan"], "Channel")

    df_band.to_csv(
        os.path.join(SAVE_DIR, f"{dataset}_Subband_Frequency.csv"),
        index=False
    )
    df_chan.to_csv(
        os.path.join(SAVE_DIR, f"{dataset}_Channel_Frequency.csv"),
        index=False
    )

    print(f"Saved {dataset} frequency tables.")
