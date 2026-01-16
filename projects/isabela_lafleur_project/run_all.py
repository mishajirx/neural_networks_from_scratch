import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from math import sqrt

from utils import (
    load_data,
    load_full_train_and_test,
    make_loaders,
    fit,
    predict,
    make_submission,
)

from model_defs import MODEL_FACTORY


def main():
    epochs = 30
    lr = 1e-3
    batch_size = 64

    # load data
    X_train, X_val, y_train, y_val, X_test, _ = load_data()
    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, y_train, y_val, X_test, batch_size=batch_size
    )

    input_dim = X_train.shape[1]
    num_classes = len(set(y_train))

    # train models
    results = []
    histories = {}
    trained_models = {}

    model_keys = ["01_linear", "02_small", "03_medium", "04_large", "05_dropout"]

    for key in model_keys:
        print(f"\n=== Training {key} ===")

        if key == "05_dropout":
            model = MODEL_FACTORY[key](input_dim, num_classes, dropout=0.3)
        else:
            model = MODEL_FACTORY[key](input_dim, num_classes)

        model, history = fit(model, train_loader, val_loader, epochs=epochs, lr=lr)

        # store
        histories[key] = history
        trained_models[key] = model

        final_val_acc = history["val_acc"][-1]
        final_val_loss = history["val_loss"][-1]

        results.append(
            {
                "model": key,
                "final_val_acc": final_val_acc,
                "final_val_loss": final_val_loss,
            }
        )

        print(f"{key} | val_acc={final_val_acc:.4f} | val_loss={final_val_loss:.4f}")

    results_df = pd.DataFrame(results).sort_values("final_val_acc", ascending=False)
    print("\n=== Summary (sorted by val accuracy) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("results.csv", index=False)
    print("\nSaved results to results.csv")

    best_key = results_df.iloc[0]["model"]
    print(f"\nBest model by val accuracy: {best_key}")

    X_full, y_full, X_test_full = load_full_train_and_test()
    full_train_loader, _, full_test_loader = make_loaders(
        X_full, X_full, y_full, y_full, X_test_full, batch_size=batch_size
    )

    if best_key == "05_dropout":
        best_model = MODEL_FACTORY[best_key](input_dim, num_classes, dropout=0.3)
    else:
        best_model = MODEL_FACTORY[best_key](input_dim, num_classes)

    best_model, _ = fit(best_model, full_train_loader, full_train_loader, epochs=epochs, lr=lr)

    test_preds = predict(best_model, full_test_loader)
    make_submission(test_preds, out_path="submission.csv")

    print("Wrote submission.csv")
    return histories, results_df



def plot_all_train_accuracy(histories):
    epoch_nums = range(1, len(next(iter(histories.values()))["train_acc"]) + 1)

    plt.figure()
    for key, history in histories.items():
        plt.plot(epoch_nums, history["train_acc"], label=f"{key} Train Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Epoch for All Models")
    plt.legend()
    plt.show()

def plot_all_validation_accuracy(histories):
    plt.figure()
    for key, history in histories.items():
        epoch_nums = range(1, len(history["val_acc"]) + 1)
        plt.plot(epoch_nums, history["val_acc"], label=f"{key} Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs Epoch for All Models")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    histories, results_df = main()
    plot_all_train_accuracy(histories)
    plot_all_validation_accuracy(histories)
    

