import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("machine_data-1.csv")

df['manufacturef'] = df['manufacturef'].str.upper()

manufacturers = df['manufacturef'].unique()

print("\n--- DATA SUMMARY ---\n")


for m in manufacturers:
    sub = df[df['manufacturef'] == m]
    print(f"Manufacturer {m}")
    print("Load range:", sub['load'].min(), "to", sub['load'].max())
    print("Time range:", sub['time'].min(), "to", sub['time'].max())
    print()


print("\n--- Expected Load (Mean) ---")
for m in manufacturers:
    sub = df[df['manufacturef'] == m]
    print(f"Manufacturer {m} mean load:", sub['load'].mean())


plt.figure()
sns.scatterplot(data=df, x="load", y="time", hue="manufacturef")
plt.title("Load vs Time Relationship")
plt.show()

corr = df[['load','time']].corr()
print("\nCorrelation between load and time:")
print(corr)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['load'], kde=True)
plt.title("Distribution of Load")

plt.subplot(1,2,2)
sns.histplot(df['time'], kde=True)
plt.title("Distribution of Time")

plt.show()


load_mu, load_std = stats.norm.fit(df['load'])
time_mu, time_std = stats.norm.fit(df['time'])

print("\nBest approximate distribution: Normal")
print("Load mean:", load_mu, "std:", load_std)
print("Time mean:", time_mu, "std:", time_std)


performance = df.groupby('manufacturef')['time'].mean()
print("\nAverage operation time by manufacturer:")
print(performance)

best = performance.idxmax()
print("\nBest manufacturer based on highest mean lifetime:", best)


plt.figure()
sns.boxplot(data=df, x='manufacturef', y='time')
plt.title("Performance Comparison (Time to Failure)")
plt.show()