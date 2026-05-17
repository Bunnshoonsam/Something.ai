import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# LOAD + CLEAN DATA
# =========================================================
df = pd.read_json("ytr.json")
df = df.dropna()

# Save max values BEFORE normalizing
max_duration  = df["video_duration"].max()
max_watchtime = df["watch_time"].max()
max_percent   = df["watch_percent"].max()

# Normalize ONCE
df["video_duration"] /= max_duration
df["watch_time"]     /= max_watchtime
df["watch_percent"]  /= max_percent

# Drop NaNs after normalization
df = df.dropna(subset=["watch_percent", "liked"])

# Clean liked column
df["liked"] = df["liked"].astype(str).str.strip().str.lower()
df["liked"] = df["liked"].replace({
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "1": 1, "0": 0,
    "1.0": 1, "0.0": 0
})
df["liked"] = pd.to_numeric(df["liked"], errors="coerce")
df = df[df["liked"].isin([0, 1])].reset_index(drop=True)

# Sanity check
assert df.isnull().sum().sum() == 0, "Still have NaNs!"
print(f"Training on {len(df)} rows")
print(f"Liked distribution:\n{df['liked'].value_counts()}")

# =========================================================
# ENCODE CATEGORICAL COLUMNS
# =========================================================
user_encoder     = LabelEncoder()
video_encoder    = LabelEncoder()
category_encoder = LabelEncoder()
device_encoder   = LabelEncoder()
time_encoder     = LabelEncoder()

df["user_encoded"]     = user_encoder.fit_transform(df["user_id"])
df["video_encoded"]    = video_encoder.fit_transform(df["video_id"])
df["category_encoded"] = category_encoder.fit_transform(df["category"])
df["device_encoded"]   = device_encoder.fit_transform(df["device"])
df["time_encoded"]     = time_encoder.fit_transform(df["watch_time_of_day"])

# =========================================================
# BUILD TENSORS
# =========================================================
user_ids      = torch.tensor(df["user_encoded"].values,  dtype=torch.long)
video_ids     = torch.tensor(df["video_encoded"].values, dtype=torch.long)
category_ids  = torch.tensor(df["category_encoded"].values, dtype=torch.long)
device_ids    = torch.tensor(df["device_encoded"].values,   dtype=torch.long)
time_ids      = torch.tensor(df["time_encoded"].values,     dtype=torch.long)

video_duration = torch.tensor(df["video_duration"].values, dtype=torch.float32).unsqueeze(1)
watch_time     = torch.tensor(df["watch_time"].values,     dtype=torch.float32).unsqueeze(1)
watch_percent  = torch.tensor(df["watch_percent"].values,  dtype=torch.float32).unsqueeze(1)

targets = torch.tensor(df["liked"].values, dtype=torch.float32).unsqueeze(1)

# =========================================================
# DATASET + DATALOADER
# =========================================================
dataset = TensorDataset(
    user_ids, video_ids, category_ids,
    device_ids, time_ids,
    video_duration, watch_time, watch_percent,
    targets
)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# =========================================================
# MODEL
# =========================================================
NUM_USERS      = df["user_encoded"].nunique()
NUM_VIDEOS     = df["video_encoded"].nunique()
NUM_CATEGORIES = df["category_encoded"].nunique()
NUM_DEVICES    = df["device_encoded"].nunique()
NUM_TIMES      = df["time_encoded"].nunique()

class YouTubeRecommender(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings
        self.user_embedding     = nn.Embedding(NUM_USERS,      64)
        self.video_embedding    = nn.Embedding(NUM_VIDEOS,     64)
        self.category_embedding = nn.Embedding(NUM_CATEGORIES,  8)
        self.device_embedding   = nn.Embedding(NUM_DEVICES,     4)
        self.time_embedding     = nn.Embedding(NUM_TIMES,       4)

        # 64 + 64 + 8 + 4 + 4 + 3 numeric = 147
        self.fc1     = nn.Linear(147, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, 32)
        self.fc4     = nn.Linear(32, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(128)
        self.bn2     = nn.BatchNorm1d(64)
        self.bn3     = nn.BatchNorm1d(32)

    def forward(self, user_id, video_id, category_id,
                device_id, time_id,
                video_duration, watch_time, watch_percent):

        user_vec     = self.user_embedding(user_id)
        video_vec    = self.video_embedding(video_id)
        category_vec = self.category_embedding(category_id)
        device_vec   = self.device_embedding(device_id)
        time_vec     = self.time_embedding(time_id)

        x = torch.cat([
            user_vec, video_vec, category_vec,
            device_vec, time_vec,
            video_duration, watch_time, watch_percent
        ], dim=1)

        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

# =========================================================
# CREATE MODEL
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = YouTubeRecommender().to(device)
print(f"Using device: {device}")

# =========================================================
# LOSS + OPTIMIZER
# =========================================================
criterion = nn.BCEWithLogitsLoss()
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

    for batch in loader:
        (user_id, video_id, category_id,
         device_id, time_id,
         vid_dur, wat_time, wat_pct, target) = [b.to(device) for b in batch]

        optimizer.zero_grad()
        prediction = model(user_id, video_id, category_id,
                           device_id, time_id,
                           vid_dur, wat_time, wat_pct)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy tracking
        predicted_labels = (torch.sigmoid(prediction) >= 0.5).float()
        correct += (predicted_labels == target).sum().item()
        total   += target.size(0)

    scheduler.step()
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1:>2}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "youtube_recommender.pth")
print("\nMODEL SAVED.")

# =========================================================
# TEST PREDICTION
# =========================================================
model.eval()
with torch.no_grad():
    sample_user     = torch.tensor([10],                              dtype=torch.long).to(device)
    sample_video    = torch.tensor([50],                              dtype=torch.long).to(device)
    sample_category = torch.tensor([1],                               dtype=torch.long).to(device)
    sample_device   = torch.tensor([0],                               dtype=torch.long).to(device)
    sample_time     = torch.tensor([2],                               dtype=torch.long).to(device)
    sample_duration = torch.tensor([[1200.0 / max_duration]],        dtype=torch.float32).to(device)
    sample_watch    = torch.tensor([[900.0  / max_watchtime]],       dtype=torch.float32).to(device)
    sample_percent  = torch.tensor([[0.75]],                         dtype=torch.float32).to(device)

    logit       = model(sample_user, sample_video, sample_category,
                        sample_device, sample_time,
                        sample_duration, sample_watch, sample_percent)
    probability = torch.sigmoid(logit).item()

print(f"\nLIKE PROBABILITY : {probability:.4f}")
print(f"PREDICTION       : {'LIKED' if probability >= 0.5 else 'NOT LIKED'}")