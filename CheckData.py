import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"StressLevelDataset.csv")

# Extract features (X) and target variable (y)
X = dataFrame[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache',
               'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 
               'living_conditions', 'safety', 'basic_needs', 'academic_performance', 
               'study_load', 'teacher_student_relationship', 'future_career_concerns', 
               'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']]

y = dataFrame['stress_level']

# Convert X to a DataFrame and plot histograms for each feature
X_df = pd.DataFrame(X, columns=X.columns)

colors = sns.color_palette("Set2")

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15), facecolor='black', gridspec_kw={'hspace': 0.5, 'wspace': 0.5}) 

# Plot histograms
for i, column in enumerate(X_df.columns):
    row, col = divmod(i, 5)
    ax = axes[row, col]
    ax.hist(X_df[column], bins=20, color='skyblue', edgecolor='white')

    # Calculate the mean, median, and mode for each column
    mean_value = X_df[column].mean()
    median_value = X_df[column].median()
    mode_value = X_df[column].mode()[0]  # Mode may have multiple values, so we take the first one

    # Add correct statistics text in a box and lines in the upper right corner
    statistics_text = f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\nMode: {mode_value}'
    ax.text(0.95, 0.95, statistics_text, transform=ax.transAxes, color='white', fontsize=6, ha='right', va='top',
            bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3'))

    # Add lines representing the mean, median, and mode
    ax.axvline(x=mean_value, color='red', linestyle='--', label='Mean')
    ax.axvline(x=median_value, color='green', linestyle='--', label='Median')
    ax.axvline(x=mode_value, color='blue', linestyle='--', label='Mode')

    ax.set_title(f'{column.capitalize()}', color='white', fontsize=8)
    ax.set_ylabel('Frequency', color='white', fontsize=8)
    ax.set_xlabel(column, color='white', fontsize=8)  # Add x-axis label
    ax.tick_params(axis='both', colors='white', labelsize=6)  # Adjust label font size
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_facecolor('black')

# Create a new figure for both pie chart and heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')

# Plot pie chart
class_distribution = y.value_counts()
percentages = class_distribution / class_distribution.sum() * 100
labels_with_percentage = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(class_distribution.index, percentages)]
ax1.pie(class_distribution, colors=sns.color_palette("Set3"), startangle=90, wedgeprops=dict(width=0.3),
        labels=labels_with_percentage, textprops={'color': 'white'})  # Set text color to white
ax1.set_title('Stress Level Distribution', color='white', fontsize=12)

# Plot correlation matrix with labels
corr_data = pd.concat([X, pd.Series(y, name='stress_level')], axis=1)
correlation_matrix = corr_data.corr()
# Plot the correlation matrix as a heatmap
heatmap = sns.heatmap(correlation_matrix[['stress_level']], annot=True, cmap='coolwarm', fmt=".2f", linewidths=2, square=False,
                      cbar_kws={"label": "Correlation"}, annot_kws={"size": 10}, ax=ax2, xticklabels=True, yticklabels=True)

# Set the color of tick labels to white
heatmap.set_xticklabels(heatmap.get_xticklabels(), color='white', fontsize=8, rotation=45, ha='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), color='white', fontsize=8, rotation=0, ha='right')

# Show the plot
plt.show()
