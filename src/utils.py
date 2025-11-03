"""
Utility functions for spam detection project.
"""

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def load_model(model_path: str):
    """
    Load a pickled model.

    Args:
        model_path: Path to .pkl file

    Returns:
        Loaded model object
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(model, save_path: str):
    """
    Save model to pickle file.

    Args:
        model: Model object to save
        save_path: Destination path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load SMS spam dataset.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with 'label' and 'message' columns
    """
    df = pd.read_csv(file_path, header=None, names=["label", "message"])
    return df


def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot distribution of spam vs ham messages.

    Args:
        df: DataFrame with 'label' column
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    counts = df["label"].value_counts()

    plt.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(counts.index)

    # Add value labels on bars
    for i, (label, count) in enumerate(counts.items()):
        plt.text(i, count + 50, str(count), ha="center", va="bottom", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Class distribution plot saved to {save_path}")

    plt.show()


def plot_text_length_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot distribution of message lengths.

    Args:
        df: DataFrame with 'label' and 'message' columns
        save_path: Optional path to save figure
    """
    df["length"] = df["message"].apply(len)

    plt.figure(figsize=(12, 5))

    # Overall distribution
    plt.subplot(1, 2, 1)
    plt.hist(df["length"], bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Message Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Overall Message Length Distribution")
    plt.grid(alpha=0.3)

    # Spam vs Ham
    plt.subplot(1, 2, 2)
    spam_lengths = df[df["label"] == "spam"]["length"]
    ham_lengths = df[df["label"] == "ham"]["length"]

    plt.hist([ham_lengths, spam_lengths], bins=30, label=["Ham", "Spam"], color=["#2ecc71", "#e74c3c"], alpha=0.7)
    plt.xlabel("Message Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Message Length by Class")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Text length distribution plot saved to {save_path}")

    plt.show()


def generate_wordcloud(texts: list, title: str = "Word Cloud", save_path: str = None, max_words: int = 100):
    """
    Generate word cloud from text data.

    Args:
        texts: List of text strings
        title: Plot title
        save_path: Optional path to save figure
        max_words: Maximum number of words to display
    """
    # Combine all texts
    combined_text = " ".join(texts)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400, max_words=max_words, background_color="white", colormap="viridis"
    ).generate(combined_text)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Word cloud saved to {save_path}")

    plt.show()


def print_dataset_info(df: pd.DataFrame):
    """
    Print comprehensive dataset information.

    Args:
        df: Dataset DataFrame
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    if "message" in df.columns:
        df["length"] = df["message"].apply(len)
        print("\nMessage Length Statistics:")
        print(df.groupby("label")["length"].describe())

    print("\nSample Messages:")
    print("\nHam examples:")
    print(df[df["label"] == "ham"]["message"].head(3).values)
    print("\nSpam examples:")
    print(df[df["label"] == "spam"]["message"].head(3).values)

    print("=" * 50)


if __name__ == "__main__":
    # Test utilities with dummy data

    # Create dummy dataset
    dummy_data = {
        "label": ["ham"] * 100 + ["spam"] * 50,
        "message": ["This is a legitimate message"] * 100 + ["Win FREE prize now!!!"] * 50,
    }
    df = pd.DataFrame(dummy_data)

    print("Testing utility functions...")
    print_dataset_info(df)

    # Test plotting
    plot_class_distribution(df)
    plot_text_length_distribution(df)
