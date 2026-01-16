from utils import load_data, make_loaders, fit
from model_defs import model_01_linear
import matplotlib.pyplot as plt
from model_defs import MODEL_FACTORY


X_train, X_val, y_train, y_val, X_test, _ = load_data()
train_loader, val_loader, _ = make_loaders(X_train, X_val, y_train, y_val, X_test)

model = model_01_linear(
    input_dim=X_train.shape[1],
    num_classes=len(set(y_train))
)

model, history = fit(model, train_loader, val_loader)

print("Final validation accuracy at last training epoch:", history["val_acc"][-1])



def plot_stuff(history):
    epoch_nums = range(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.plot(epoch_nums, history["train_acc"], label="Train Accuracy")
    plt.plot(epoch_nums, history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    epochs = 30
    lr = 1e-3
    batch_size = 64

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

    model_key = "01_linear"
    print(f"\n=== Training {model_key} ===")
    model = MODEL_FACTORY[model_key](input_dim, num_classes)
    model, history = fit(model, train_loader, val_loader, epochs=epochs, lr=lr)
    plot_stuff(history)
