import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import fashion_mnist
print("Starting...")
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
print("Dataset loaded!")
X_train_full = X_train_full.reshape(60000, 784).astype("float32") / 255.0
X_test_full = X_test_full.reshape(10000, 784).astype("float32") / 255.0
X_train = X_train_full[:10000]
y_train = y_train_full[:10000]
X_test = X_test_full[:2000]
y_test = y_test_full[:2000]
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
k_values = [1, 3, 5, 7, 9]
accuracies = []
times = []
for k in k_values:
    print(f"\nRunning KNN with K = {k}")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    start = time.time()
    pred = model.predict(X_test)
    elapsed = time.time() - start
    accuracy = accuracy_score(y_test, pred) * 100
    accuracies.append(accuracy)
    times.append(elapsed)
    print(f"Accuracy = {accuracy:.2f}%")
    print(f"Time = {elapsed:.3f} seconds")
print("\n-----------------------------")
print("K Value | Accuracy | Time")
print("-----------------------------")
for k, acc, t in zip(k_values, accuracies, times):
    print(f"{k:^7} | {acc:7.2f}% | {t:.3f}s")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, "bo-")
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs K")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(k_values, times, "rs-")
plt.xlabel("K")
plt.ylabel("Time (seconds)")
plt.title("Time vs K")
plt.grid(True)
plt.tight_layout()
plt.show()