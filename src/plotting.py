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

def plot_predictors(df):
    features = ['SP_density', 'beta_rest', 'theta_power', 'SO_SP_coupling']
    for feat in features:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=feat, y='behavior', hue='day')
        plt.title(f"Behavior vs {feat}")
        plt.tight_layout()
        plt.savefig(f"outputs/behavior_vs_{feat}.png")
