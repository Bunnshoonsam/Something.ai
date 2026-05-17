import random

# =========================
# LOAD SENTENCES
# =========================

with open("sentences.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# =========================
# PHRASE REPLACEMENTS
# =========================

replacements = {

    "deeply immersed in the content": [
        "highly engaged with the content",
        "showing strong viewer retention",
        "actively consuming the content",
        "highly attentive to the content",
    ],

    "low engagement": [
        "weak audience engagement",
        "reduced viewer interaction",
        "limited audience response",
        "below-average engagement",
    ],

    "strong completion": [
        "excellent watch completion",
        "high viewer retention",
        "strong viewing consistency",
        "solid completion performance",
    ],

    "click-through rate": [
        "CTR",
        "viewer click rate",
        "recommendation click performance",
    ],

    "subscribe at": [
        "convert into subscribers at",
        "show subscription conversion at",
        "subscribe at approximately",
    ]
}

# =========================
# EXPAND SENTENCES
# =========================

expanded = []

for line in lines:

    expanded.append(line)

    for key, values in replacements.items():

        if key in line:

            for v in values:

                new_line = line.replace(key, v)

                expanded.append(new_line)

# =========================
# SHUFFLE
# =========================

random.shuffle(expanded)

# =========================
# REMOVE DUPLICATES
# =========================

expanded = list(set(expanded))

# =========================
# SAVE
# =========================

with open("expanded_sentences.txt", "w", encoding="utf-8") as f:

    for line in expanded:

        f.write(line + "\n")

print("Original:", len(lines))
print("Expanded:", len(expanded))

print("\nSample:\n")

for s in expanded[:20]:

    print("-", s)