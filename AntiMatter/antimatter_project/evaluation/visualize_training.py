import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_metrics(log_file="../model/training_log.csv", output_dir="../model/plots"):
    """
    Reads the training log CSV and generates professional quality training curves
    for presentation/reporting.
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(log_file)
    
    # Set Style
    sns.set_theme(style="darkgrid")
    
    # 1. Loss Curve
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="loss", linewidth=2.5, color="#e74c3c")
    plt.title("Antimatter-300M: Training Loss Convergence", fontsize=16, pad=20)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.axhline(y=df['loss'].min(), color='green', linestyle='--', alpha=0.5, label=f"Min Loss: {df['loss'].min():.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300)
    print(f"Saved loss curve to {output_dir}/loss_curve.png")
    
    # 2. Perplexity Curve
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="perplexity", linewidth=2.5, color="#8e44ad")
    plt.yscale("log")
    plt.title("Antimatter-300M: Perplexity Reduction (Log Scale)", fontsize=16, pad=20)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/perplexity_curve.png", dpi=300)
    print(f"Saved perplexity curve to {output_dir}/perplexity_curve.png")

    # 3. Learning Rate Schedule
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="step", y="lr", linewidth=2.0, color="#2980b9")
    plt.title("Cosine Learning Rate Schedule", fontsize=14)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lr_schedule.png", dpi=300)
    print(f"Saved LR schedule to {output_dir}/lr_schedule.png")

if __name__ == "__main__":
    plot_training_metrics()
