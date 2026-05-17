import numpy as np
import pandas as pd
#coloumn = name,age,gender,sleep_hours,screen_time_hours,exercise_hours,stress_level,social_media_hours,mood_score,behaviour_type
mapping = {
    "Healthy": 5,
    "Balanced": 4,
    "Moderate": 3,
    "Stressed": 1,
    "Anxious": 2
}
df=pd.read_csv("YTR.csv")

print(df["liked"].unique())
df = df.dropna(subset=["liked"])

df["liked"] = df["liked"].replace({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

df["liked"] = pd.to_numeric(
    df["liked"],
    errors="coerce"
)

df = df[df["liked"].isin([0,1])]
# dc = pd.DataFrame(
#     dr["behaviour_type"]
# )
# dr["encoded"]=dr["behaviour_type"].map(mapping)
# print(dr[["name","behaviour_type","encoded"]])

# encoded = pd.get_dummies(dc)
# print(dc.shape)
# print(encoded)
# print(dr.groupby('behaviour_type')
# ['stress_level'].agg(['mean','std','min','max']))

# a=dr["behaviour_type","name"].astype("category").cat.codes
# print(a)
# print(dr.columns)
#print(dr.describe)
#print(dr["age"]>23)
# a= np.array([[2,4,5],[3,6,7],[6,34,5]])
# b= np.array([[0,4,5],[3,9,2],[4,5,6]])
# c=a.flatten()
# d=b.reshape(9,1)
# print(a.shape)
# print(b.shape)
# print(np.dot(c,d))

# print(np.random.rand(2,3))
# print(np.random.choice(a.flatten(),4))
# np.random.shuffle(a)
# print(a)
# np.random.seed(56)
# print(np.random.rand(4))


# #a=a.reshape(1,9)
# print(a.shape)
# print(a)
# print(a.ndim)


# d  = np.array([[[1, 2], [3, 4]],
#               [[5, 6], [7, 8]]],dtype=float)


# shape: (2, 2, 2)
# print(d.shape)

# print(d.ndim)
# print(d[1][1][0])
# print(dtype(d))
# print(d[0][0])
# print(d[0][0][0])
# print(d[1,1,0])



