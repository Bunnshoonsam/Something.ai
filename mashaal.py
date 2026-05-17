import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# your encoders
behaviour_encoder = {
    'Healthy':  5,
    'Balanced': 4,
    'Moderate': 3,
    'Stressed': 1,
    'Anxious':  2
}

gender_encoder = {
    'Male':   1,
    'Female': 0
}

# load CSV
df = pd.read_csv('data.csv')

# apply encoders to actual columns
df['behaviour_encoded'] = df['behaviour_type'].map(behaviour_encoder)
df['gender_encoded'] = df['gender'].map(gender_encoder)

# features and target using real columnimport torch.utils.data from Dataset names
X = df[['age', 'sleep_hours', 'screen_time_hours',
        'exercise_hours', 'stress_level',
        'social_media_hours', 'gender_encoded']]

Y = df['behaviour_encoded']

# convert to tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor  = torch.tensor(Y.values, dtype=torch.long)

# dataset and loader loading datasets
dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=8, shuffle=True)

print("Dataset size:", len(dataset))
print("X shape:", X_tensor.shape)
print("y shape:", y_tensor.shape)

#here we go
#suoer u know a universal variable in class self is first 
# layers all we use matrix multiplaicatoin weigth and bias using a 
#usinf a formula y = wx+b (weight matrix and bias vector)Then why 16 specifically?

# general rule of thumb:
# - start bigger than input
# - powers of 2 are preferred
#   4, 8, 16, 32, 64, 128...
#   because GPU memory works efficiently with powers of 2

# 7 inputs → 16 is a safe starting guess
# not too small → model can learn patterns
# not too big   → won't overfit on 30 rows

class BehaviourModel(nn.Module):

    def __init__(self):
        super().__init__()             # 👈 NEVER skip this
        self.layer1 = nn.Linear(7, 16) # 7 features → 16 neurons
        self.layer2 = nn.Linear(16, 8) # 16 → 8 neurons
        self.layer3 = nn.Linear(8, 6)  # 8 → 5 outputs (5 behaviour types)
        self.relu   = nn.ReLU()        # activation function

    def forward(self, x):              # 👈 data flows through here
        x = self.relu(self.layer1(x)) # layer 1 → activate
        x = self.relu(self.layer2(x)) # layer 2 → activate
        x = self.layer3(x)            # final output, no relu here
        return x

model = BehaviourModel()

# ── 6. LOSS AND OPTIMIZER ────────────────────────────
criterion = nn.CrossEntropyLoss()              # for multi-class (5 types) calculating the loss 
optimizer = torch.optim.Adam(model.parameters(), lr=1)#learnig rate can affect the output

# ── 7. TRAINING LOOP ─────────────────────────────────
for epoch in range(100):

    for X_batch, y_batch in loader:            # batch by batch

        prediction = model(X_batch)            # forward pass → predict

        loss = criterion(prediction, y_batch)  # how wrong are we

        optimizer.zero_grad()                  # clear old gradients 👈
        loss.backward()                        # backpropagation magic 👈
        optimizer.step()                       # update weights 👈

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")


def reality_check(person_data, predicted_behaviour):

    age, sleep, screen, exercise, stress, social, gender = person_data

    print(f"\n{'='*40}")
    print(f"  REALITY CHECK — You are: {predicted_behaviour}")
    print(f"{'='*40}")

    # sleep check
    if sleep < 6:
        print(f"😴 Sleep: {sleep}hrs — CRITICAL. Minimum 7hrs needed.")
    elif sleep < 7:
        print(f"😴 Sleep: {sleep}hrs — Low. Try to get 7-8hrs.")
    else:
        print(f"✅ Sleep: {sleep}hrs — Good.")

    # screen time check
    if screen > 8:
        print(f"📱 Screen: {screen}hrs — Way too high. Max 4hrs recommended.")
    elif screen > 5:
        print(f"📱 Screen: {screen}hrs — High. Reduce to under 5hrs.")
    else:
        print(f"✅ Screen: {screen}hrs — Good.")

    # exercise check
    if exercise == 0:
        print(f"🏃 Exercise: NONE — This is destroying your mental health.")
    elif exercise < 1:
        print(f"🏃 Exercise: {exercise}hrs — Very low. Aim for 1hr daily.")
    else:
        print(f"✅ Exercise: {exercise}hrs — Good.")

    # stress check
    if stress >= 8:
        print(f"😤 Stress: {stress}/10 — DANGER ZONE. Immediate action needed.")
    elif stress >= 5:
        print(f"😤 Stress: {stress}/10 — Moderate. Work on reducing.")
    else:
        print(f"✅ Stress: {stress}/10 — Good.")

    # social media check
    if social > 5:
        print(f"📲 Social Media: {social}hrs — Addiction level. Cut to 2hrs.")
    elif social > 3:
        print(f"📲 Social Media: {social}hrs — High. Reduce to under 2hrs.")
    else:
        print(f"✅ Social Media: {social}hrs — Good.")

    # final recommendation based on behaviour
    print(f"\n{'='*40}")
    print("  YOUR ACTION PLAN:")
    print(f"{'='*40}")

    if predicted_behaviour == 'Anxious':
        print("1. Sleep before 11pm — non negotiable")
        print("2. No phone 1hr before bed")
        print("3. 20min walk every morning")
        print("4. Reduce social media to 1hr max")
        print("5. Write 3 things you're grateful for daily")

    elif predicted_behaviour == 'Stressed':
        print("1. You are overloaded — remove one commitment")
        print("2. Exercise is mandatory — even 15min walk counts")
        print("3. Sleep 8hrs — stress without sleep = burnout")
        print("4. Talk to someone — friend, family, anyone")
        print("5. Screen time is making it worse — digital detox")

    elif predicted_behaviour == 'Moderate':
        print("1. You're okay but could be better")
        print("2. Add 30min exercise to your routine")
        print("3. Cut screen time by 1hr daily")
        print("4. Improve sleep consistency — same time every day")

    elif predicted_behaviour == 'Balanced':
        print("1. You're doing well — maintain consistency")
        print("2. Push exercise to 1.5hrs for even better results")
        print("3. Watch your screen time — don't let it creep up")

    elif predicted_behaviour == 'Healthy':
        print("1. EXCELLENT — you are a role model")
        print("2. Help someone around you improve their habits")
        print("3. Keep the consistency — don't slip on weekends")

# ── 8. TEST A PREDICTION ─────────────────────────────
model.eval()

# keep as list first
person = [22, 5, 8, 0.5, 8, 5, 1]
#          age sleep screen ex stress social gender

# tensor for model
Shiwang = torch.tensor([person], dtype=torch.float32)

# predict
output          = model(Shiwang)
prediction      = torch.argmax(output)
reverse_encoder = {v: k for k, v in behaviour_encoder.items()}#do as the name do nothing complex 

predicted_label = reverse_encoder[prediction.item()]  # clean word only

# print and check
print(f"\nPredicted behaviour: {predicted_label}")
reality_check(person, predicted_label)  # list + clean word ✅