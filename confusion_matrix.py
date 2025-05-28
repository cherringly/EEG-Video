import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

labels = ['Positive', 'Negative']
# cm = np.array([[81.76, 18.24],
#                 [6.45, 93.55]])
# cm = np.array([[88.60,11.40],
#                [14.15,85.85]])
cm = np.array([[163, 23],
                [28, 1911]])

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100


plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 20})

plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Confusion Matrix for EOG Detector (%)', fontsize=16)


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()