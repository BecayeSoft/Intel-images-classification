import Data

from CNN import CNN
from EfficientNet import EfficientNet

# --------------------------
# Load the data
# --------------------------
train_data, val_data, test_data, classes = Data.load_data()
x_test, y_test = Data.get_x_and_y(data=test_data)  # separate images from labels
Data.show_samples(data=train_data, classes=classes)

# --------------------------
# CNN
# --------------------------
cnn = CNN()
cnn.build(version=2)
cnn.compile()

# Load weights - Already trained
cnn.load_weights()
# cnn.fit(train_data=train_data, val_data=val_data, batch_size=32, epochs=15, patience=2)
# cnn.save()

# Compute performance metrics
cnn.compute_metrics(x_test, y_test)
print(f"Kappa Score: {cnn.metrics.get('kappa')}")
print(f"Classification Report\m {cnn.metrics.get('report')}")

# Visualize performances
cnn.plot_history()

# ----------------------------
# EfficientNet
# ----------------------------
enet = EfficientNet()
enet.build(version=2)
enet.unfreeze()
enet.compile()
enet.fit(train_data=train_data, val_data=val_data, epochs=15, patience=2)

enet.save()

# Compute performance metrics
# enet.compute_metrics(x=x_test, y=y_test)

# Visualize performances
enet.plot_history()
