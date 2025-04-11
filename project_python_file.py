### EDA ###
#---------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")

print(df.columns)                  # Column names
print(df.shape)                    # Number of rows and columns
print(df.info())                   # Data types and non-null counts
print(df.describe)                 # Summary for numerical columns
print(df.head())                   # First five records
print(df.tail())                   # Last five records
print(df.isnull().sum())           # Total missing values per column
print(df.duplicated().sum())       # Check for duplicate rows
print(df.dropna())                 # Remove missing/duplicate values
print(df.fillna(method='ffill'))   # Fill missing/duplicate values





### Objective 1 ###
# --------------------------------------------------------------------------------------------------------------
# To generate descriptive statistics and identify key trends, distributions, and inconsistencies in 
# sector-wise revenue expenditure across states and union territories.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")

sector_columns = data.columns[3:78]

# 1. Generate basic summary statistics
summary = data[sector_columns].describe()
print("Sector-wise Descriptive Statistics:\n", summary)

# 2. Analyze missing values
missing_info = data[sector_columns].isnull().sum()
missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
print("\nMissing Value Count by Sector:\n", missing_info)

# 3. Visualize distributions of selected expenditure sectors
selected = [
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities",
    "Energy",
    "Transport and communications"
]

data[selected].hist(bins=25, figsize=(8, 6), color='red', edgecolor='black')
plt.suptitle("Distribution of Expenditures in Key Sectors")
plt.tight_layout()
plt.show()

# 5. Trend analysis: Mean total expenditure by year
data["srcYear"] = data["srcYear"].astype(str)
avg_expenditure_by_year = data.groupby("srcYear")["Total expenditure"].mean()

plt.figure(figsize=(8, 6))
sns.lineplot(x=avg_expenditure_by_year.index, y=avg_expenditure_by_year.values, marker='o', color='blue')
plt.title("Mean Total Expenditure Across Financial Years")
plt.xlabel("Financial Year")
plt.ylabel("Average Expenditure")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()





### Objective 2 ###
#---------------------------------------------------------------------------------------------------------------
## To design and implement visual representations using Matplotlib and Seaborn for 
## comparing sector-wise allocations across years and regions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")

# Filter to use 'Accounts' for consistency
filtered_data = df[df["Budget type"] == "Accounts"].copy()

# Ensure 'srcYear' is string type for plotting
filtered_data["srcYear"] = filtered_data["srcYear"].astype(str)

# Set a few important sectors for visual comparison
important_sectors = [
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities"
]

# 1. Barplot: Year-wise sector expenditure

sector_yearly = filtered_data.groupby("srcYear")[important_sectors].mean()
sector_yearly.plot(kind='bar', figsize=(12, 6), colormap="Set2")
plt.title("Average Sector-wise Expenditure Over Years")
plt.ylabel("Average Expenditure")
plt.xlabel("Financial Year")
plt.show()

# 3. Lineplot: Trend in one sector over years across states

selected_states = ['MAHARASHTRA', 'BIHAR', 'KARNATAKA', 'TAMIL NADU']
plt.figure(figsize=(8, 6))
for state in selected_states:
    state_data = filtered_data[filtered_data["srcStateName"] == state]
    sns.lineplot(
        x="srcYear",
        y="Education, sports, art and culture",
        data=state_data,
        label=state,
        marker='o'
    )

plt.title("Education Sector Trend Over Time (Selected States)")
plt.xlabel("Financial Year")
plt.ylabel("Expenditure")
plt.xticks(rotation=45)
plt.legend(title="State")
plt.grid(True)
plt.tight_layout()
plt.show()




### Objective 3 ###
#-------------------------------------------------------------------------------------------------------------------
## To conduct statistical hypothesis testing to evaluate significant differences in 
## revenue expenditure patterns between selected sectors or regions.

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")
df = df[df["Budget type"] == "Accounts"]

# Drop rows with missing data for target sectors
df_clean = df.dropna(subset=[
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities"
])

# 1. T-Test: Education vs Medical Expenditure
edu = df_clean["Education, sports, art and culture"]
med = df_clean["Medical and public health"]

t_stat1, p_val1 = stats.ttest_ind(edu, med, equal_var=False)
print("T-Test: Education vs Medical Expenditure")
print(f"T-Statistic: {t_stat1:.3f}, P-Value: {p_val1:.4f}")
if p_val1 < 0.05:
    print("Significant difference exists between education and medical sector spending.\n")
else:
    print("No significant difference found between education and medical sector spending.\n")

# Boxplot: Education vs Medical
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean[["Education, sports, art and culture", "Medical and public health"]], orient='h')
plt.title("Boxplot: Education vs Medical Expenditure")
plt.xlabel("Expenditure")
plt.yticks(ticks=[0, 1], labels=["Education", "Medical"])
plt.tight_layout()
plt.show()

# 2. T-Test: Agriculture in Tamil Nadu vs Karnataka
agri1 = df_clean[df_clean["srcStateName"] == "TAMIL NADU"]["Agriculture and allied activities"]
agri2 = df_clean[df_clean["srcStateName"] == "KARNATAKA"]["Agriculture and allied activities"]

t_stat2, p_val2 = stats.ttest_ind(agri1, agri2, equal_var=False)
print("T-Test: Agriculture Expenditure - Tamil Nadu vs Karnataka")
print(f"T-Statistic: {t_stat2:.3f}, P-Value: {p_val2:.4f}")
if p_val2 < 0.05:
    print("Significant difference in agriculture spending between Tamil Nadu and Karnataka.\n")
else:
    print("No significant difference in agriculture spending between the two states.\n")






### Objective 4 ###
#------------------------------------------------------------------------------------------------------------------
## To uncover relationships and correlations among sectors based on 
## revenue allocation patterns through numerical and visual statistical analysis.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")
df = df[df["Budget type"] == "Accounts"]

sectors_to_analyze = [
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities",
    "Energy",
    "Transport and communications",
    "Social security and welfare",
    "Urban development"
]

# Drop rows with missing values for the selected sectors
df_corr = df[sectors_to_analyze].dropna()

# 1. Compute the correlation matrix
correlation_matrix = df_corr.corr()
print("Correlation Matrix:\n")
print(correlation_matrix)

# 2. Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Sector-wise Revenue Allocation")
plt.tight_layout()
plt.show()

# 3. Pairplot (optional for deep exploration)
sns.pairplot(df_corr)
plt.suptitle("Pairwise Sector Relationships", y=1.02)
plt.show()

# 4. Find top 3 positively and negatively correlated pairs
# Flatten matrix, remove self-correlations, sort
corr_pairs = correlation_matrix.unstack().reset_index()
corr_pairs.columns = ['Sector 1', 'Sector 2', 'Correlation']
filtered_pairs = corr_pairs[corr_pairs['Sector 1'] != corr_pairs['Sector 2']]

# Drop duplicate pairs
filtered_pairs['pair_key'] = filtered_pairs[['Sector 1', 'Sector 2']].apply(lambda row: tuple(sorted(row)), axis=1)
filtered_pairs = filtered_pairs.drop_duplicates('pair_key').drop(columns='pair_key')

# Top correlations
print("\nTop 3 Positive Correlations:")
print(filtered_pairs.sort_values(by='Correlation', ascending=False).head(3))

print("\nTop 3 Negative Correlations:")
print(filtered_pairs.sort_values(by='Correlation', ascending=True).head(3))




### Objective 5 ###
#----------------------------------------------------------------------------------------------------------------
## To analyze year-on-year changes in sector-wise revenue expenditure to 
## detect growth patterns, stagnation, or sudden shifts in government spending priorities.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\shubh\Downloads\7567_all_files\ZIP\7567\7567_source_data.csv")
df = df[df["Budget type"] == "Accounts"]
df["srcYear"] = df["srcYear"].astype(str)

# Choose sectors to analyze
sectors = [
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities",
    "Energy",
    "Transport and communications"
]

# Group by year and calculate average expenditure
yearly_avg = df.groupby("srcYear")[sectors].mean()

# Compute year-on-year % change
yearly_pct_change = yearly_avg.pct_change() * 100

# 1. Line plots for actual year-on-year expenditure
plt.figure(figsize=(12, 6))
for sector in sectors:
    sns.lineplot(x=yearly_avg.index, y=yearly_avg[sector], label=sector)

plt.title("Year-on-Year Average Expenditure by Sector")
plt.xlabel("Financial Year")
plt.ylabel("Average Expenditure")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Line plots for % change (growth/stagnation detection)
plt.figure(figsize=(12, 6))
for sector in sectors:
    sns.lineplot(x=yearly_pct_change.index, y=yearly_pct_change[sector], label=sector)

plt.title("Year-on-Year % Change in Expenditure by Sector")
plt.xlabel("Financial Year")
plt.ylabel("Percent Change (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Identify major shifts (jumps/drops)
print("\nSudden Year-on-Year Changes:")
for sector in sectors:
    big_changes = yearly_pct_change[abs(yearly_pct_change[sector]) > 20][sector]
    if not big_changes.empty:
        print(f"\n{sector}:")
        print(big_changes.round(2))
