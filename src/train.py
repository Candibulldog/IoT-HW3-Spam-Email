"""
Enhanced training script that saves comprehensive evaluation data for visualization.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import evaluate_model
from src.preprocess import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedSpamClassifierTrainer:
    """Enhanced trainer with visualization data export."""

    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = TextPreprocessor(max_features=3000)
        self.models = {}
        self.results = {}
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Load and prepare dataset."""
        logger.info(f"Loading data from {self.data_path}")

        df = pd.read_csv(self.data_path, header=None, names=["label", "message"])

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Class distribution:\n{df['label'].value_counts()}")

        df["label"] = df["label"].map({"ham": 0, "spam": 1})

        return df

    def prepare_features(self, df: pd.DataFrame):
        """Preprocess and split data."""
        logger.info("Preprocessing text data...")

        cleaned_texts, features = self.preprocessor.fit_transform(df["message"].values)
        labels = df["label"].values

        logger.info(f"Feature matrix shape: {features.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )

        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes."""
        logger.info("Training Naive Bayes...")
        start_time = time.time()

        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(f"Naive Bayes training completed in {elapsed:.2f}s")

        return model

    def train_logistic_regression(self, X_train, y_train, max_iter: int = 1000):
        """Train Logistic Regression."""
        logger.info("Training Logistic Regression...")
        start_time = time.time()

        model = LogisticRegression(max_iter=max_iter, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(f"Logistic Regression training completed in {elapsed:.2f}s")

        return model

    def train_random_forest(self, X_train, y_train, n_estimators: int = 100):
        """Train Random Forest."""
        logger.info("Training Random Forest...")
        start_time = time.time()

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(f"Random Forest training completed in {elapsed:.2f}s")

        return model

    def train_svm(self, X_train, y_train, kernel: str = "linear"):
        """Train SVM."""
        logger.info("Training SVM...")
        start_time = time.time()

        model = SVC(kernel=kernel, probability=True, random_state=self.random_state)
        model.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(f"SVM training completed in {elapsed:.2f}s")

        return model

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate."""
        self.models["naive_bayes"] = self.train_naive_bayes(X_train, y_train)
        self.models["logistic_regression"] = self.train_logistic_regression(X_train, y_train)
        self.models["random_forest"] = self.train_random_forest(X_train, y_train)
        self.models["svm"] = self.train_svm(X_train, y_train)

        logger.info("\n" + "=" * 50)
        logger.info("Model Evaluation Results")
        logger.info("=" * 50)

        for name, model in self.models.items():
            logger.info(f"\n{name.upper()}:")
            metrics = evaluate_model(model, X_test, y_test)
            self.results[name] = metrics

            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        best_model = self.models[best_model_name]

        logger.info("\n" + "=" * 50)
        logger.info(f"Best model: {best_model_name.upper()}")
        logger.info(f"F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        logger.info("=" * 50)

        return best_model, best_model_name

    def save_evaluation_data(self, output_dir: str = "models"):
        """Save comprehensive evaluation data for visualization."""
        logger.info("Saving evaluation data for visualization...")

        os.makedirs(output_dir, exist_ok=True)

        # 1. Save metrics comparison data
        metrics_df = pd.DataFrame(
            [
                {
                    "Model": name.replace("_", " ").title(),
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1_score"],
                    "ROC-AUC": metrics["roc_auc"],
                }
                for name, metrics in self.results.items()
            ]
        )

        metrics_path = os.path.join(output_dir, "metrics_comparison.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics comparison saved to {metrics_path}")

        # 2. Save confusion matrix for best model
        best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        best_model = self.models[best_model_name]

        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        cm_path = os.path.join(output_dir, "confusion_matrix.npy")
        np.save(cm_path, cm)
        logger.info(f"Confusion matrix saved to {cm_path}")

        # 3. Save ROC curve data
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)

            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc_score(self.y_test, y_pred_proba)),
            }

            roc_path = os.path.join(output_dir, "roc_curve_data.json")
            with open(roc_path, "w") as f:
                json.dump(roc_data, f)
            logger.info(f"ROC curve data saved to {roc_path}")

        # 4. Generate and save visualization plots
        self.generate_plots(output_dir)

    def generate_plots(self, output_dir: str):
        """Generate and save evaluation plots."""
        logger.info("Generating evaluation plots...")

        # Plot 1: Metrics comparison
        metrics_df = pd.DataFrame(
            [
                {
                    "Model": name.replace("_", " ").title(),
                    "Accuracy": metrics["accuracy"],
                    "F1-Score": metrics["f1_score"],
                }
                for name, metrics in self.results.items()
            ]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics_df))
        width = 0.35

        ax.bar(x - width / 2, metrics_df["Accuracy"], width, label="Accuracy", color="#3498db")
        ax.bar(x + width / 2, metrics_df["F1-Score"], width, label="F1-Score", color="#e74c3c")

        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
        ax.legend()
        ax.set_ylim([0.9, 1.0])
        ax.grid(axis="y", alpha=0.3)

        plot_path = os.path.join(output_dir, "metrics_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metrics plot saved to {plot_path}")
        plt.close()

        # Plot 2: Confusion Matrix
        best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        best_model = self.models[best_model_name]

        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            ax=ax,
            annot_kws={"size": 16},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {best_model_name.replace('_', ' ').title()}")

        cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {cm_plot_path}")
        plt.close()

    def save_model(self, model, model_name: str, output_dir: str = "models"):
        """Save model, vectorizer, and metadata."""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, "spam_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        # Save vectorizer
        vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.preprocessor.vectorizer, f)
        logger.info(f"Vectorizer saved to {vectorizer_path}")

        # Save metadata
        metadata = {
            "model_name": model_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "metrics": self.results[model_name],
        }

        metadata_path = os.path.join(output_dir, "model_metadata.txt")
        with open(metadata_path, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Metadata saved to {metadata_path}")

        # Save as JSON too for easier parsing
        json_path = os.path.join(output_dir, "model_metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata JSON saved to {json_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train spam detection models (Enhanced)")
    parser.add_argument("--data", type=str, default="data/sms_spam_no_header.csv", help="Path to dataset CSV")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")

    args = parser.parse_args()

    # Initialize trainer
    trainer = EnhancedSpamClassifierTrainer(
        data_path=args.data, test_size=args.test_size, random_state=args.random_state
    )

    # Load and prepare data
    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_features(df)

    # Train all models
    best_model, best_model_name = trainer.train_all_models(X_train, X_test, y_train, y_test)

    # Save model and evaluation data
    trainer.save_model(best_model, best_model_name, args.output_dir)
    trainer.save_evaluation_data(args.output_dir)

    logger.info("\n✅ Training completed successfully!")
    logger.info(f"📊 Evaluation data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
