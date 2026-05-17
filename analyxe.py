import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_json("ytr.json")

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"]  = df["timestamp"].dt.dayofweek

# =========================================================
# CLEAN BOOLEAN-LIKE COLUMNS
# (liked, commented, subscribed_after, recommended, clicked
#  may be True/False strings or mixed)
# =========================================================
bool_cols = ["liked", "commented", "subscribed_after", "recommended", "clicked"]
for col in bool_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": 1, "false": 0, "yes": 1, "no": 0,
                  "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# NORMALIZE NUMERICAL COLUMNS
# =========================================================
df["video_duration"] = df["video_duration"] / df["video_duration"].max()
df["watch_time"]     = df["watch_time"]     / df["watch_time"].max()
df["watch_percent"]  = df["watch_percent"]  / df["watch_percent"].max()

# =========================================================
# DROP NaNs
# =========================================================
numerical_cols = [
    "video_duration", "watch_time", "liked", "commented",
    "subscribed_after", "recommended", "watch_percent", "hour", "day"
]
df = df.dropna(subset=numerical_cols + ["clicked"]).reset_index(drop=True)

# Sanity check
assert df[numerical_cols].isnull().sum().sum() == 0, "Still have NaNs in numerical cols!"
assert df["clicked"].isnull().sum() == 0, "Still have NaNs in target!"
print(f"Clean dataset: {len(df)} rows")
print(f"Click distribution:\n{df['clicked'].value_counts()}")

# =========================================================
# ENCODE CATEGORICAL DATA
# =========================================================
user_encoder     = LabelEncoder()
video_encoder    = LabelEncoder()
category_encoder = LabelEncoder()
device_encoder   = LabelEncoder()
time_encoder     = LabelEncoder()

df["user_id"]          = user_encoder.fit_transform(df["user_id"])
df["video_id"]         = video_encoder.fit_transform(df["video_id"])
df["category"]         = category_encoder.fit_transform(df["category"])
df["device"]           = device_encoder.fit_transform(df["device"])
df["watch_time_of_day"]= time_encoder.fit_transform(df["watch_time_of_day"])

# Store counts before building tensors
NUM_USERS      = df["user_id"].nunique()
NUM_VIDEOS     = df["video_id"].nunique()
NUM_CATEGORIES = df["category"].nunique()
NUM_DEVICES    = df["device"].nunique()
NUM_TIMES      = df["watch_time_of_day"].nunique()

# =========================================================
# BUILD TENSORS
# =========================================================
user_ids   = torch.tensor(df["user_id"].values,           dtype=torch.long)
video_ids  = torch.tensor(df["video_id"].values,          dtype=torch.long)
categories = torch.tensor(df["category"].values,          dtype=torch.long)
devices    = torch.tensor(df["device"].values,            dtype=torch.long)
times      = torch.tensor(df["watch_time_of_day"].values, dtype=torch.long)

numerical = torch.tensor(
    df[numerical_cols].values.astype(np.float32),  # force float32 explicitly
    dtype=torch.float32
)

targets = torch.tensor(
    df["clicked"].values.astype(np.float32),
    dtype=torch.float32
).unsqueeze(1)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
dataset = TensorDataset(
    user_ids, video_ids, categories,
    devices, times, numerical, targets
)

train_size    = int(0.8 * len(dataset))
test_size     = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False)

# =========================================================
# MODEL
# =========================================================
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_videos, num_categories, num_devices, num_times):
        super().__init__()

        self.user_embedding     = nn.Embedding(num_users,      32)
        self.video_embedding    = nn.Embedding(num_videos,     32)
        self.category_embedding = nn.Embedding(num_categories,  8)
        self.device_embedding   = nn.Embedding(num_devices,     4)
        self.time_embedding     = nn.Embedding(num_times,       4)

        # 32+32+8+4+4 = 80 embeddings + 9 numerical = 89
        self.fc1     = nn.Linear(89, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(128)
        self.bn2     = nn.BatchNorm1d(64)

    def forward(self, user_id, video_id, category, device, time, numerical):
        x = torch.cat([
            self.user_embedding(user_id),
            self.video_embedding(video_id),
            self.category_embedding(category),
            self.device_embedding(device),
            self.time_embedding(time),
            numerical
        ], dim=1)

        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)   # raw logit — BCEWithLogitsLoss handles sigmoid
        return x

# =========================================================
# MODEL INIT
# =========================================================
device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderModel(NUM_USERS, NUM_VIDEOS, NUM_CATEGORIES, NUM_DEVICES, NUM_TIMES).to(device_name)
print(f"Using: {device_name}")

# =========================================================
# LOSS + OPTIMIZER
# =========================================================
criterion = nn.BCEWithLogitsLoss()   # more stable than BCELoss + sigmoid
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =========================================================
# TRAINING LOOP
# =========================================================
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch in train_loader:
        user_id, video_id, category, device, time, num, target = [
            b.to(device_name) for b in batch
        ]

        optimizer.zero_grad()
        pred = model(user_id, video_id, category, device, time, num)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += ((torch.sigmoid(pred) >= 0.5).float() == target).sum().item()
        total      += target.size(0)

    scheduler.step()
    print(f"Epoch {epoch+1:>2}/{epochs} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Accuracy: {100*correct/total:.2f}%")

# =========================================================
# EVALUATE ON TEST SET
# =========================================================
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for batch in test_loader:
        user_id, video_id, category, device, time, num, target = [
            b.to(device_name) for b in batch
        ]
        pred    = model(user_id, video_id, category, device, time, num)
        correct += ((torch.sigmoid(pred) >= 0.5).float() == target).sum().item()
        total   += target.size(0)

print(f"\nTest Accuracy: {100*correct/total:.2f}%")

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "recommender.pth")
print("MODEL SAVED.")

# =========================================================
# SAMPLE PREDICTION
# =========================================================
model.eval()
sample = test_dataset[0]
with torch.no_grad():
    logit = model(
        sample[0].unsqueeze(0).to(device_name),
        sample[1].unsqueeze(0).to(device_name),
        sample[2].unsqueeze(0).to(device_name),
        sample[3].unsqueeze(0).to(device_name),
        sample[4].unsqueeze(0).to(device_name),
        sample[5].unsqueeze(0).to(device_name)
    )
    prob = torch.sigmoid(logit).item()

print(f"\nClick Probability : {prob:.4f}")
print(f"Prediction        : {'CLICK' if prob >= 0.5 else 'NO CLICK'}")