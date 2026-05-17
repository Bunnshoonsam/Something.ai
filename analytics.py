import pandas as pd
import numpy as np
import random

# =========================================================
# LOAD CLEAN DATA
# =========================================================
df = pd.read_json("ytr_clean.json")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]     = df["timestamp"].dt.hour
df["day_name"] = df["timestamp"].dt.day_name()

def label_time(hour):
    if 5  <= hour < 12: return "Morning"
    if 12 <= hour < 17: return "Afternoon"
    if 17 <= hour < 21: return "Evening"
    return "Night"

df["time_slot"] = df["hour"].apply(label_time)

print(f"Loaded {len(df)} rows")
print(f"Categories: {sorted(df['category'].unique())}")
print(f"Devices:    {sorted(df['device'].unique())}")

sentences = []
random.seed(42)

categories = sorted(df["category"].unique())
devices    = sorted(df["device"].unique())
time_slots = ["Morning", "Afternoon", "Evening", "Night"]
days       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def eng(v):
    if v >= 0.65: return "very high"
    if v >= 0.50: return "high"
    if v >= 0.35: return "moderate"
    if v >= 0.20: return "low"
    return "very low"

def watch(v):
    if v >= 0.75: return "excellent"
    if v >= 0.60: return "strong"
    if v >= 0.40: return "average"
    return "poor"

def retention(v):
    if v >= 0.75: return "retains viewers exceptionally well"
    if v >= 0.60: return "holds viewer attention well"
    if v >= 0.40: return "shows average viewer retention"
    return "struggles to retain viewers"

def compare(a, b):
    d = a - b
    if d >  0.15: return "significantly outperforms"
    if d >  0.05: return "outperforms"
    if d < -0.15: return "significantly underperforms compared to"
    if d < -0.05: return "underperforms compared to"
    return "performs similarly to"

def loyalty(sub, like):
    if sub > like + 0.05: return "converts viewers to subscribers more than it generates likes"
    if like > sub + 0.05: return "generates more likes than subscriptions"
    return "balances likes and subscriptions evenly"

# Paraphrase pools
def click_p(cat, v):
    e, p = eng(v), f"{v*100:.1f}%"
    return random.choice([
        f"{cat} achieves a {e} click-through rate of {p}.",
        f"The click-through rate for {cat} stands at {p}, which is {e}.",
        f"{cat} content draws clicks at a {e} rate of {p}.",
        f"Users click on {cat} recommendations {p} of the time.",
        f"{cat} has a {p} CTR, reflecting {e} viewer interest.",
    ])

def watch_p(cat, v):
    w, p = watch(v), f"{v*100:.1f}%"
    return random.choice([
        f"{cat} shows {w} watch completion at {p} on average.",
        f"On average, viewers complete {p} of {cat} videos.",
        f"{cat} {retention(v)}, with {p} average completion.",
        f"Watch completion for {cat} is {p}, rated {w}.",
        f"{cat} content holds attention through {p} of its runtime on average.",
    ])

def like_p(cat, v):
    e, p = eng(v), f"{v*100:.1f}%"
    return random.choice([
        f"{cat} generates a {e} like rate of {p}.",
        f"Viewers like {cat} content {p} of the time.",
        f"The like rate for {cat} is {p}, showing {e} appreciation.",
        f"{cat} earns likes from {p} of its viewers.",
        f"{p} of {cat} viewers express approval through likes.",
    ])

def sub_p(cat, v):
    p = f"{v*100:.1f}%"
    return random.choice([
        f"{cat} drives subscription conversion at {p} rate.",
        f"{p} of {cat} viewers subscribe after watching.",
        f"{cat} converts {p} of its audience into subscribers.",
        f"Subscription rate after watching {cat} is {p}.",
        f"{cat} earns new subscribers from {p} of viewers.",
    ])

def comment_p(cat, v):
    p    = f"{v*100:.1f}%"
    mood = "active community discussion" if v > 0.3 else "passive viewing"
    return random.choice([
        f"{cat} has a comment rate of {p}, indicating {mood}.",
        f"{p} of {cat} viewers leave comments.",
        f"{cat} sparks {mood} with a {p} comment rate.",
        f"Viewer commentary on {cat} content sits at {p}.",
    ])

# =========================================================
# BLOCK 1 — CATEGORY OVERVIEW
# =========================================================
cat_stats = df.groupby("category").agg(
    views        = ("clicked",          "count"),
    click_rate   = ("clicked",          "mean"),
    like_rate    = ("liked",            "mean"),
    comment_rate = ("commented",        "mean"),
    sub_rate     = ("subscribed_after", "mean"),
    watch_pct    = ("watch_percent",    "mean"),
    watch_time   = ("watch_time",       "mean"),
    rec_rate     = ("recommended",      "mean"),
).round(4)

total_views = cat_stats["views"].sum()

for cat, row in cat_stats.iterrows():
    share = row["views"] / total_views
    cr    = row["click_rate"]
    wp    = row["watch_pct"]
    lr    = row["like_rate"]
    sr    = row["sub_rate"]
    cmr   = row["comment_rate"]
    wt    = row["watch_time"]
    rr    = row["rec_rate"]

    for _ in range(4):
        sentences.append(click_p(cat, cr))
        sentences.append(watch_p(cat, wp))
        sentences.append(like_p(cat, lr))
        sentences.append(sub_p(cat, sr))
        sentences.append(comment_p(cat, cmr))

    sentences.append(f"{cat} accounts for {share*100:.1f}% of total platform views.")
    sentences.append(f"Out of all categories, {cat} represents {share*100:.1f}% of viewing sessions.")
    sentences.append(f"{cat} users spend an average of {wt/60:.1f} minutes per session.")
    sentences.append(f"Average session length for {cat} viewers is {wt/60:.1f} minutes.")
    sentences.append(f"{cat} {loyalty(sr, lr)}.")

    if rr > 0.6:
        sentences.append(f"{cat} is heavily driven by the recommendation algorithm.")
        sentences.append(f"Most {cat} views come from algorithmic recommendations rather than direct search.")
    else:
        sentences.append(f"{cat} viewers mostly find content through direct search, not recommendations.")
        sentences.append(f"{cat} has lower recommendation dependency, indicating strong organic discovery.")

    if cr >= 0.65:
        sentences.append(f"{cat} is one of the top performing categories for click-through rate.")
        sentences.append(f"Content creators in {cat} benefit from very strong algorithmic visibility.")
    elif cr < 0.25:
        sentences.append(f"{cat} struggles to convert impressions into clicks.")
        sentences.append(f"Low CTR in {cat} suggests thumbnails or titles are not compelling enough.")

    if wp >= 0.70:
        sentences.append(f"{cat} content is highly compelling, keeping viewers engaged through most of the video.")
        sentences.append(f"Viewers rarely abandon {cat} videos early, reflecting strong content quality.")
    elif wp < 0.35:
        sentences.append(f"{cat} has difficulty keeping viewers engaged beyond the first portion of videos.")
        sentences.append(f"High dropout rate in {cat} suggests content may not meet viewer expectations.")

    if sr >= 0.40:
        sentences.append(f"{cat} is a powerful subscription driver on the platform.")
        sentences.append(f"Creators in {cat} grow their subscriber base faster than most categories.")

    if lr > cmr:
        sentences.append(f"{cat} viewers prefer to express appreciation silently through likes rather than comments.")
    else:
        sentences.append(f"{cat} sparks conversation, with comment rates exceeding like rates.")

# =========================================================
# BLOCK 2 — DEVICE ANALYSIS
# =========================================================
dev_stats = df.groupby("device").agg(
    views      = ("clicked",          "count"),
    click_rate = ("clicked",          "mean"),
    like_rate  = ("liked",            "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    watch_time = ("watch_time",       "mean"),
    sub_rate   = ("subscribed_after", "mean"),
).round(4)

for dev, row in dev_stats.iterrows():
    cr, wp, wt, sr, lr = (
        row["click_rate"], row["watch_pct"],
        row["watch_time"], row["sub_rate"], row["like_rate"]
    )
    for _ in range(4):
        sentences.append(random.choice([
            f"{dev} users have a {eng(cr)} click-through rate of {cr*100:.1f}%.",
            f"On {dev}, viewers click recommendations {cr*100:.1f}% of the time.",
            f"Click-through rate on {dev} is {cr*100:.1f}%, rated {eng(cr)}.",
            f"{dev} achieves a {cr*100:.1f}% CTR across all content.",
        ]))
        sentences.append(random.choice([
            f"{dev} users show {watch(wp)} watch completion at {wp*100:.1f}%.",
            f"Watch completion on {dev} averages {wp*100:.1f}%, which is {watch(wp)}.",
            f"{dev} viewers complete {wp*100:.1f}% of videos on average.",
            f"Average completion rate on {dev} is {wp*100:.1f}%.",
        ]))

    sentences.append(f"{dev} users average {wt/60:.1f} minutes per session.")
    sentences.append(f"{dev} drives {sr*100:.1f}% subscription conversion.")
    sentences.append(f"Like rate among {dev} users is {lr*100:.1f}%.")

    if wt == dev_stats["watch_time"].max():
        sentences.append(f"{dev} users spend the most time watching of any device.")
        sentences.append(f"Among all devices, {dev} produces the longest viewing sessions.")
    if wp == dev_stats["watch_pct"].max():
        sentences.append(f"{dev} leads all devices in watch completion percentage.")
        sentences.append(f"Videos are most completely watched on {dev}.")
    if cr == dev_stats["click_rate"].max():
        sentences.append(f"{dev} is the highest clicking device on the platform.")
    if cr == dev_stats["click_rate"].min():
        sentences.append(f"{dev} has the lowest click-through rate, suggesting passive viewing behavior.")
        sentences.append(f"Users on {dev} are least likely to click on recommendations.")

# =========================================================
# BLOCK 3 — TIME OF DAY
# =========================================================
time_stats = df.groupby("time_slot").agg(
    views      = ("clicked",          "count"),
    click_rate = ("clicked",          "mean"),
    like_rate  = ("liked",            "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    sub_rate   = ("subscribed_after", "mean"),
).round(4)

for time, row in time_stats.iterrows():
    cr, wp, lr, sr = (
        row["click_rate"], row["watch_pct"],
        row["like_rate"],  row["sub_rate"]
    )
    for _ in range(4):
        sentences.append(random.choice([
            f"During {time}, users show {eng(cr)} engagement with {cr*100:.1f}% CTR.",
            f"{time} sessions produce a {cr*100:.1f}% click-through rate.",
            f"Viewer click rate during {time} is {cr*100:.1f}%, which is {eng(cr)}.",
            f"{time} achieves {cr*100:.1f}% CTR across all categories.",
        ]))
        sentences.append(random.choice([
            f"{time} sessions show {watch(wp)} watch completion at {wp*100:.1f}%.",
            f"Viewers during {time} complete {wp*100:.1f}% of videos on average.",
            f"Watch completion during {time} is {wp*100:.1f}%, rated {watch(wp)}.",
            f"Average completion during {time} is {wp*100:.1f}%.",
        ]))

    sentences.append(f"Like rate during {time} is {lr*100:.1f}%.")
    sentences.append(f"Subscription conversion during {time} stands at {sr*100:.1f}%.")
    sentences.append(f"{time} attracts {row['views']:,} total viewing sessions in the dataset.")

    if cr == time_stats["click_rate"].max():
        sentences.append(f"{time} is the peak engagement window with the highest CTR of the day.")
        sentences.append(f"Content published targeting {time} viewers sees the strongest click performance.")
        sentences.append(f"Creators should schedule uploads before {time} to maximize click-through rate.")
    if wp == time_stats["watch_pct"].max():
        sentences.append(f"{time} produces the most complete viewing sessions of any time period.")
        sentences.append(f"Viewers are most focused and attentive during {time}.")
    if row["views"] == time_stats["views"].max():
        sentences.append(f"{time} is the busiest period of the day with the most total viewers.")
        sentences.append(f"More users watch content during {time} than any other time of day.")
    if cr == time_stats["click_rate"].min():
        sentences.append(f"{time} is the weakest engagement period, with distracted or casual viewing.")
        sentences.append(f"Recommendations perform least effectively during {time}.")

# =========================================================
# BLOCK 4 — CATEGORY × DEVICE
# =========================================================
cat_dev = df.groupby(["category", "device"]).agg(
    click_rate = ("clicked",          "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    like_rate  = ("liked",            "mean"),
    sub_rate   = ("subscribed_after", "mean"),
    views      = ("clicked",          "count"),
).round(4)

for (cat, dev), row in cat_dev.iterrows():
    if row["views"] < 50: continue
    cr, wp, sr, lr = (
        row["click_rate"], row["watch_pct"],
        row["sub_rate"],   row["like_rate"]
    )
    for _ in range(3):
        sentences.append(random.choice([
            f"{cat} on {dev} has {eng(cr)} engagement with {cr*100:.1f}% CTR.",
            f"On {dev}, {cat} content achieves a {cr*100:.1f}% click-through rate.",
            f"{dev} users click on {cat} recommendations {cr*100:.1f}% of the time.",
            f"{cat} content on {dev} attracts clicks at {cr*100:.1f}% rate.",
        ]))
        sentences.append(random.choice([
            f"{cat} on {dev} achieves {watch(wp)} completion at {wp*100:.1f}%.",
            f"{dev} users complete {wp*100:.1f}% of {cat} videos on average.",
            f"Watch completion for {cat} on {dev} is {wp*100:.1f}%, rated {watch(wp)}.",
            f"{cat} viewed on {dev} has {wp*100:.1f}% average completion.",
        ]))

    sentences.append(f"{cat} on {dev} drives {sr*100:.1f}% subscription rate.")
    sentences.append(f"{cat} content receives {lr*100:.1f}% like rate on {dev}.")

    if cr >= 0.60:
        sentences.append(f"{cat} performs exceptionally well on {dev} and should be prioritized for this platform.")
        sentences.append(f"Content creators in {cat} should focus on {dev} for maximum reach and engagement.")
        sentences.append(f"The {cat} and {dev} combination is one of the strongest engagement pairs on the platform.")
    if wp >= 0.70:
        sentences.append(f"{dev} users are deeply committed to {cat} content, watching most of each video.")
        sentences.append(f"{cat} on {dev} produces some of the most complete viewing sessions on the platform.")
    if cr < 0.25:
        sentences.append(f"{cat} struggles to attract clicks from {dev} users, indicating a poor content fit.")
        sentences.append(f"Creators should reconsider {dev} strategy for {cat} due to consistently low engagement.")
    if sr >= 0.45:
        sentences.append(f"{cat} on {dev} is a powerful subscription funnel.")
        sentences.append(f"Nearly half of all {cat} viewers on {dev} become subscribers.")

# =========================================================
# BLOCK 5 — CATEGORY × TIME
# =========================================================
cat_time = df.groupby(["category", "time_slot"]).agg(
    click_rate = ("clicked",          "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    like_rate  = ("liked",            "mean"),
    sub_rate   = ("subscribed_after", "mean"),
    views      = ("clicked",          "count"),
).round(4)

for (cat, time), row in cat_time.iterrows():
    if row["views"] < 50: continue
    cr, wp, lr, sr = (
        row["click_rate"], row["watch_pct"],
        row["like_rate"],  row["sub_rate"]
    )
    for _ in range(3):
        sentences.append(random.choice([
            f"{cat} during {time} has {eng(cr)} click-through rate of {cr*100:.1f}%.",
            f"During {time}, {cat} content achieves {cr*100:.1f}% CTR.",
            f"{time} is a {eng(cr)} engagement window for {cat} with {cr*100:.1f}% CTR.",
            f"{cat} attracts {cr*100:.1f}% CTR during {time} sessions.",
        ]))
        sentences.append(random.choice([
            f"{cat} viewed during {time} shows {watch(wp)} completion at {wp*100:.1f}%.",
            f"During {time}, viewers complete {wp*100:.1f}% of {cat} videos.",
            f"{cat} watch completion during {time} is {wp*100:.1f}%, rated {watch(wp)}.",
            f"Average completion for {cat} during {time} is {wp*100:.1f}%.",
        ]))

    sentences.append(f"{cat} during {time} generates {lr*100:.1f}% like rate.")
    sentences.append(f"Viewers of {cat} during {time} subscribe at {sr*100:.1f}% rate.")

    if cr >= 0.60:
        sentences.append(f"{cat} during {time} is a peak engagement window and ideal publishing time.")
        sentences.append(f"Publishing {cat} content during {time} maximizes click-through performance.")
        sentences.append(f"Creators in {cat} should prioritize {time} uploads for best results.")
    if wp >= 0.70:
        sentences.append(f"{time} is the best time for {cat} as viewers watch nearly the entire video.")
        sentences.append(f"{cat} viewers during {time} are among the most engaged on the platform.")
    if cr < 0.25:
        sentences.append(f"{cat} underperforms during {time}, suggesting viewers are not receptive then.")
        sentences.append(f"Avoid publishing {cat} content during {time} due to consistently low engagement.")
    if sr > lr:
        sentences.append(f"{cat} during {time} converts viewers to subscribers more than it generates likes.")

# =========================================================
# BLOCK 6 — TRIPLE COMBO (category × device × time)
# =========================================================
triple = df.groupby(["category", "device", "time_slot"]).agg(
    click_rate = ("clicked",          "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    sub_rate   = ("subscribed_after", "mean"),
    views      = ("clicked",          "count"),
).round(4)

for (cat, dev, time), row in triple.iterrows():
    if row["views"] < 30: continue
    cr, wp, sr = row["click_rate"], row["watch_pct"], row["sub_rate"]

    for _ in range(2):
        sentences.append(random.choice([
            f"{cat} on {dev} during {time} has {eng(cr)} engagement at {cr*100:.1f}% CTR.",
            f"During {time}, {cat} content on {dev} achieves {cr*100:.1f}% click rate.",
            f"{dev} users watching {cat} during {time} click at {cr*100:.1f}% rate.",
        ]))
        sentences.append(random.choice([
            f"{cat} on {dev} during {time} achieves {watch(wp)} completion of {wp*100:.1f}%.",
            f"{dev} viewers complete {wp*100:.1f}% of {cat} videos during {time}.",
            f"Watch completion for {cat} on {dev} during {time} is {wp*100:.1f}%.",
        ]))

    sentences.append(
        f"{cat} on {dev} during {time} converts {sr*100:.1f}% of viewers to subscribers."
    )

    if cr >= 0.60:
        sentences.append(f"The combination of {cat}, {dev}, and {time} is a goldmine for platform engagement.")
        sentences.append(f"Targeting {cat} on {dev} during {time} delivers maximum click performance.")
    if wp >= 0.72:
        sentences.append(f"{dev} users watching {cat} during {time} are deeply immersed in the content.")
    if cr < 0.22:
        sentences.append(f"{cat} on {dev} during {time} is one of the weakest performing combinations.")
        sentences.append(f"The {cat}, {dev}, {time} combination consistently underperforms and should be avoided.")

# =========================================================
# BLOCK 7 — DAY OF WEEK
# =========================================================
day_stats = df.groupby("day_name").agg(
    click_rate = ("clicked",          "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    like_rate  = ("liked",            "mean"),
    sub_rate   = ("subscribed_after", "mean"),
    views      = ("clicked",          "count"),
).round(4)

weekday_avg = day_stats[
    day_stats.index.isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])
]["watch_pct"].mean()

for day, row in day_stats.iterrows():
    cr, wp, lr, sr = (
        row["click_rate"], row["watch_pct"],
        row["like_rate"],  row["sub_rate"]
    )
    for _ in range(2):
        sentences.append(random.choice([
            f"On {day}, users show {eng(cr)} engagement with {cr*100:.1f}% click rate.",
            f"{day} produces a {cr*100:.1f}% click-through rate, rated {eng(cr)}.",
            f"Click-through rate on {day} is {cr*100:.1f}%.",
            f"{day} achieves {cr*100:.1f}% CTR across all content.",
        ]))

    sentences.append(f"{day} produces {watch(wp)} watch completion at {wp*100:.1f}%.")
    sentences.append(f"Like rate on {day} is {lr*100:.1f}%.")
    sentences.append(f"Subscription conversion on {day} stands at {sr*100:.1f}%.")
    sentences.append(f"{day} records {row['views']:,} total viewing sessions.")

    if day in ["Saturday", "Sunday"]:
        if wp > weekday_avg:
            sentences.append(f"Weekend viewing on {day} shows deeper engagement than average weekday sessions.")
            sentences.append(f"{day} viewers have more free time, resulting in more complete viewing sessions.")
        else:
            sentences.append(f"Despite more free time, {day} viewers show lower completion than weekday users.")
            sentences.append(f"{day} engagement is surprisingly weaker than typical weekday performance.")

    if cr == day_stats["click_rate"].max():
        sentences.append(f"{day} achieves the highest click rate of the week.")
        sentences.append(f"Content performs best when published on {day} based on click-through data.")
    if wp == day_stats["watch_pct"].max():
        sentences.append(f"{day} produces the most complete viewing sessions of any day of the week.")
    if row["views"] == day_stats["views"].max():
        sentences.append(f"{day} is the busiest day with the highest total viewership.")
        sentences.append(f"More content is consumed on {day} than any other day of the week.")

# =========================================================
# BLOCK 8 — RECOMMENDED VS ORGANIC
# =========================================================
rec_stats = df.groupby("recommended").agg(
    click_rate = ("clicked",          "mean"),
    like_rate  = ("liked",            "mean"),
    watch_pct  = ("watch_percent",    "mean"),
    sub_rate   = ("subscribed_after", "mean"),
).round(4)

if len(rec_stats) == 2:
    org = rec_stats.iloc[0]
    rec = rec_stats.iloc[1]

    for _ in range(3):
        sentences.append(random.choice([
            f"Recommended videos have {rec['click_rate']*100:.1f}% CTR vs {org['click_rate']*100:.1f}% for organic.",
            f"Algorithm-recommended content achieves {rec['click_rate']*100:.1f}% CTR vs {org['click_rate']*100:.1f}% organic.",
            f"Organic discovery yields {org['click_rate']*100:.1f}% CTR while recommendations yield {rec['click_rate']*100:.1f}%.",
            f"Recommended content outperforms organic by {abs(rec['click_rate']-org['click_rate'])*100:.1f} percentage points in CTR.",
        ]))

    sentences.append(f"Recommended content achieves {rec['watch_pct']*100:.1f}% watch completion vs {org['watch_pct']*100:.1f}% organic.")
    sentences.append(f"Subscription rate for recommended content is {rec['sub_rate']*100:.1f}% vs {org['sub_rate']*100:.1f}% organic.")
    sentences.append(f"Like rate for recommended content is {rec['like_rate']*100:.1f}% vs {org['like_rate']*100:.1f}% organic.")

    if rec["click_rate"] > org["click_rate"]:
        diff = (rec["click_rate"] - org["click_rate"]) * 100
        sentences.append(f"The recommendation algorithm improves click-through rate by {diff:.1f} percentage points.")
        sentences.append(f"Recommended videos significantly outperform organic discovery in driving clicks.")
        sentences.append(f"Optimizing for algorithmic recommendation is a key strategy for maximizing engagement.")
        sentences.append(f"Creators benefit strongly from appearing in the recommendation feed.")
    else:
        diff = (org["click_rate"] - rec["click_rate"]) * 100
        sentences.append(f"Organic content outperforms recommendations by {diff:.1f} percentage points in CTR.")
        sentences.append(f"Users who actively search for content are more engaged than passive recommendation viewers.")
        sentences.append(f"Building strong search presence may be more valuable than chasing algorithm recommendations.")

# =========================================================
# BLOCK 9 — CATEGORY COMPARISONS
# =========================================================
cat_list = list(cat_stats.index)
for i in range(len(cat_list)):
    for j in range(i+1, len(cat_list)):
        a, b   = cat_list[i], cat_list[j]
        ac, bc = cat_stats.loc[a,"click_rate"], cat_stats.loc[b,"click_rate"]
        aw, bw = cat_stats.loc[a,"watch_pct"],  cat_stats.loc[b,"watch_pct"]
        as_, bs = cat_stats.loc[a,"sub_rate"],  cat_stats.loc[b,"sub_rate"]
        al, bl = cat_stats.loc[a,"like_rate"],  cat_stats.loc[b,"like_rate"]

        sentences.append(f"{a} {compare(ac, bc)} {b} in click-through rate ({ac*100:.1f}% vs {bc*100:.1f}%).")
        sentences.append(f"{a} {compare(aw, bw)} {b} in watch completion ({aw*100:.1f}% vs {bw*100:.1f}%).")

        if as_ > bs:
            sentences.append(f"{a} is a stronger subscription driver than {b} ({as_*100:.1f}% vs {bs*100:.1f}%).")
        else:
            sentences.append(f"{b} drives more subscriptions than {a} ({bs*100:.1f}% vs {as_*100:.1f}%).")

        if al > bl:
            sentences.append(f"{a} earns more viewer appreciation than {b} ({al*100:.1f}% vs {bl*100:.1f}% like rate).")
        else:
            sentences.append(f"{b} is more liked than {a} ({bl*100:.1f}% vs {al*100:.1f}%).")

# =========================================================
# BLOCK 10 — WATCH COMPLETION BUCKETS
# =========================================================
df["watch_bucket"] = pd.cut(
    df["watch_percent"],
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=["0-25%", "25-50%", "50-75%", "75-100%"]
)

bucket_cat = df.groupby(["category", "watch_bucket"], observed=True).agg(
    click_rate = ("clicked",          "mean"),
    sub_rate   = ("subscribed_after", "mean"),
    like_rate  = ("liked",            "mean"),
    views      = ("clicked",          "count"),
).round(4)

for (cat, bucket), row in bucket_cat.iterrows():
    if row["views"] < 30: continue
    cr, sr, lr = row["click_rate"], row["sub_rate"], row["like_rate"]
    sentences.append(f"{cat} viewers who watch {bucket} of a video click at {cr*100:.1f}% rate.")
    sentences.append(f"{cat} viewers completing {bucket} of a video subscribe at {sr*100:.1f}% rate.")
    sentences.append(f"{cat} viewers who reach {bucket} completion like content at {lr*100:.1f}% rate.")

    if bucket == "75-100%":
        if sr > 0.40:
            sentences.append(f"{cat} viewers who finish nearly the entire video are highly likely to subscribe.")
            sentences.append(f"Completing a {cat} video is a strong predictor of subscription behavior.")
        if cr > 0.55:
            sentences.append(f"{cat} viewers who watch most of a video are very likely to click the next recommendation.")
    if bucket == "0-25%":
        if cr < 0.25:
            sentences.append(f"Users who barely start a {cat} video rarely engage with future recommendations.")
            sentences.append(f"Early dropout in {cat} strongly signals low future engagement.")

# =========================================================
# BLOCK 11 — CREATOR ADVICE
# =========================================================
for cat, row in cat_stats.iterrows():
    cr, wp, sr, cmr = (
        row["click_rate"], row["watch_pct"],
        row["sub_rate"],   row["comment_rate"]
    )
    sentences.append(f"Content creators in {cat} should focus on the first 30 seconds to reduce early dropout.")
    sentences.append(f"Thumbnail and title optimization is critical for {cat} given its {eng(cr)} click rate.")
    sentences.append(f"To grow in {cat}, creators should publish consistently to build algorithmic momentum.")
    sentences.append(f"Creators in {cat} should study top performing videos to understand what drives high watch completion.")

    if wp < 0.45:
        sentences.append(f"{cat} creators should improve content pacing to reduce the high viewer dropout rate.")
        sentences.append(f"Shorter video formats may perform better in {cat} due to low average completion.")
    if sr > 0.40:
        sentences.append(f"{cat} creators should include strong calls to action since the audience converts well.")
        sentences.append(f"End screens and subscribe prompts work especially well for {cat} creators.")
    if cmr > 0.25:
        sentences.append(f"{cat} creators should actively respond to comments to nurture their highly engaged community.")
        sentences.append(f"Community engagement through comments is a key growth lever for {cat} creators.")

for dev, row in dev_stats.iterrows():
    sentences.append(f"Creators should ensure {dev}-optimized formatting and thumbnails for {dev} viewers.")
    if row["watch_pct"] > 0.60:
        sentences.append(f"{dev} is a high-value platform for creators due to its strong watch completion rates.")
        sentences.append(f"Investing in {dev} audience growth pays off with higher retention and engagement.")

for time, row in time_stats.iterrows():
    cr = row["click_rate"]
    if cr == time_stats["click_rate"].max():
        sentences.append(f"Creators should schedule video releases before {time} to capture peak engagement.")
        sentences.append(f"The optimal upload window is before {time} when viewer click intent is highest.")
    if cr == time_stats["click_rate"].min():
        sentences.append(f"Avoid scheduling major releases during {time} due to consistently low engagement.")
        sentences.append(f"{time} is the worst time to publish new content based on engagement data.")

# =========================================================
# BLOCK 12 — PLATFORM SUMMARY INSIGHTS
# =========================================================
best_cat_click = cat_stats["click_rate"].idxmax()
worst_cat_click= cat_stats["click_rate"].idxmin()
best_cat_watch = cat_stats["watch_pct"].idxmax()
best_cat_sub   = cat_stats["sub_rate"].idxmax()
best_cat_like  = cat_stats["like_rate"].idxmax()
best_dev       = dev_stats["click_rate"].idxmax()
best_dev_watch = dev_stats["watch_pct"].idxmax()
best_time      = time_stats["click_rate"].idxmax()
best_time_watch= time_stats["watch_pct"].idxmax()

sentences.append(f"Across all categories, {best_cat_click} leads the platform in click-through rate.")
sentences.append(f"{worst_cat_click} is the lowest performing category by click-through rate.")
sentences.append(f"{best_cat_watch} produces the highest quality viewing sessions with best watch completion.")
sentences.append(f"{best_cat_sub} is the strongest subscription driver, converting viewers to loyal subscribers.")
sentences.append(f"{best_cat_like} generates the most viewer appreciation through likes.")
sentences.append(f"{best_dev} is the most valuable device for content engagement on this platform.")
sentences.append(f"{best_dev_watch} leads all devices in watch completion percentage.")
sentences.append(f"{best_time} is the optimal time window for content discovery and engagement.")
sentences.append(f"{best_time_watch} produces the most attentive and engaged viewing sessions.")
sentences.append(f"The platform sees strongest engagement during {best_time} on {best_dev} devices.")
sentences.append(f"To maximize reach, creators should target {best_cat_click} content on {best_dev} during {best_time}.")
sentences.append(f"The highest performing niche is {best_cat_click} viewed on {best_dev} during {best_time}.")
sentences.append(f"Platform-wide click-through rate is {df['clicked'].mean()*100:.1f}%.")
sentences.append(f"Platform-wide average watch completion is {df['watch_percent'].mean()*100:.1f}%.")
sentences.append(f"Platform-wide subscription conversion rate is {df['subscribed_after'].mean()*100:.1f}%.")
sentences.append(f"Platform-wide like rate stands at {df['liked'].mean()*100:.1f}%.")
sentences.append(f"The platform has {df['user_id'].nunique():,} unique users and {df['video_id'].nunique():,} unique videos.")
sentences.append(f"Total viewing sessions analyzed: {len(df):,}.")

# =========================================================
# SAVE
# =========================================================
sentences = list(dict.fromkeys(s.strip() for s in sentences if s.strip()))
random.shuffle(sentences)

with open("sentences.txt", "w") as f:
    for s in sentences:
        f.write(s + "\n")

print(f"\nTotal sentences : {len(sentences)}")
print(f"Saved to        : sentences.txt")
print(f"\nSample:")
print("-" * 55)
for s in sentences[:15]:
    print(s)