import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/512/label_infos.csv')

# Plotting the histogram
plt.hist(df['ink_p'], bins=20, edgecolor='black')
plt.xlabel('Ink Percentage')
plt.ylabel('Frequency')
plt.title('Histogram of Ink Percentage')

# Printing statistics
print("Total patches: ", df.shape[0])
print("Total patches with ink percentage > 0: ", df[df['ink_p'] > 0].shape[0])
print("Total patches with ink percentage > 3: ", df[df['ink_p'] > 3].shape[0])
# ink_stats = df['ink_p'].describe()
# print(ink_stats)
plt.show()
