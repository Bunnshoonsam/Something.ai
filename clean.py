import pandas as pd

df = pd.read_json("ytr.json")

# =========================================================
# FIX 1: CLEAN DIRTY CATEGORIES
# =========================================================
category_map = {
    "MUsic":   "Music",
    "music":   "Music",
    "gamingg": "Gaming",
    "Ed":      "Education",
    "COMEDY":  "Comedy",
}
df["category"] = df["category"].str.strip().replace(category_map)

# =========================================================
# FIX 2: CLEAN BOOLEAN COLUMNS
# =========================================================
bool_cols = ["liked", "commented", "subscribed_after", "recommended", "clicked"]
for col in bool_cols:
    df[col] = (
        df[col].astype(str).str.strip().str.lower()
        .replace({"true": 1, "false": 0, "yes": 1, "no": 0,
                  "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# FIX 3: CLEAN WATCH PERCENT (remove impossible values)
# =========================================================
df["watch_percent"] = pd.to_numeric(df["watch_percent"], errors="coerce")
df = df[df["watch_percent"].between(0, 1)]   # only keep 0-100% values

# =========================================================
# FIX 4: CLEAN WATCH TIME AND DURATION (remove outliers)
# =========================================================
df["watch_time"]      = pd.to_numeric(df["watch_time"],      errors="coerce")
df["video_duration"]  = pd.to_numeric(df["video_duration"],  errors="coerce")

# Remove impossible watch times (more than 24 hours)
df = df[df["watch_time"].between(0, 86400)]
df = df[df["video_duration"].between(0, 86400)]

# =========================================================
# DROP NULLS
# =========================================================
df = df.dropna(subset=bool_cols + ["watch_percent", "watch_time", 
                                    "video_duration", "category", "device"])
df = df.reset_index(drop=True)

# =========================================================
# VERIFY
# =========================================================
print("Categories:", sorted(df["category"].unique()))
print("Devices:   ", sorted(df["device"].unique()))
print("Rows:      ", len(df))
print("\nwatch_percent range:", df["watch_percent"].min(), "to", df["watch_percent"].max())
print("watch_time range:   ", df["watch_time"].min(),    "to", df["watch_time"].max())
print("\nNull counts:")
print(df.isnull().sum())

# =========================================================
# SAVE CLEAN DATA
# =========================================================
df.to_json("ytr_clean.json", orient="records")
print("\nSaved clean data to ytr_clean.json")