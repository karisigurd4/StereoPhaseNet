import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
log_data = pd.read_csv("training_log.csv")

# Plot the training loss
plt.figure(figsize=(12, 6))
plt.plot(log_data["Epoch"], log_data["Training Loss"], label='Training Loss')
plt.plot(log_data["Epoch"], log_data["Phase Coherence"], label='Phase Coherence')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Phase Coherence Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
