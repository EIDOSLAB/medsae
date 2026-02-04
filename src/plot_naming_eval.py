import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config
import os

file = '/scratch/medical-mi/medclip-rn50_chexpert_standard_sweep_best_eval_results_accuracy.csv'
# Load your CSV files
df1 = pd.read_csv('/scratch/medical-mi/medclip-rn50_chexpert_standard_sweep_best_random_eval_results.csv.csv')
df2 = pd.read_csv('/scratch/medical-mi/MedCLIPONLY_eval_results.csv.csv')
df3 = pd.read_csv(file)

df1['accuracy'] = df1['accuracy'].str.replace(',', '.', regex=False)
df2['accuracy'] = df2['accuracy'].str.replace(',', '.', regex=False)
df3['accuracy'] = df3['accuracy'].str.replace(',', '.', regex=False)

df1['accuracy'] = df1['accuracy'].astype(float)
df2['accuracy'] = df2['accuracy'].astype(float)
df3['accuracy'] = df3['accuracy'].astype(float)

# Add a label to each DataFrame to indicate the model
df1['Model'] = 'Random explanation'
df2['Model'] = 'MedCLIP raw embeddings'
df3['Model'] = 'SAE features'

# Concatenate the DataFrames
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# Plot density plot using seaborn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_all, x='accuracy', hue='Model', fill=True, common_norm=False, alpha=0.3, linewidth=2)

# Zoom in on the x-axis (accuracy > 0.6)
plt.xlim(0.2, 0.85)
# plt.ylim(0, 0.2)


# Customize plot
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Feature naming', labels = ['SAE features', 'MedCLIP raw embeddings', 'Random explanation'], fontsize=12, title_fontsize=14)



plt.show()

plt.savefig(os.path.join(config.RESULTS, 'naming_eval_accuracy.png'), dpi=400, bbox_inches='tight')