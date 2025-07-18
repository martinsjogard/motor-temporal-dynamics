# src/plotting.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curve(df):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='day', y='behavior', ci='sd', estimator='mean')
    plt.title("Behavioral Performance Across Days")
    plt.xlabel("Day")
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.savefig("outputs/learning_curve.png")
