# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Feature Selection, Model Selection, And Tuning Project
# ### Github: https://github.com/cmelende/FeatureAndModelSelectionAndTuningProject.git
# ### Cory Melendez
# ### 8/28/2020

# %%
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from univariateAnalysis import UniVariateAnalysis, UniVariateReport

sns.set(rc={'figure.figsize':(11.7,8.27)})
concrete_df = pd.read_csv('data/concrete.csv')

columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
targetColumn = 'strength'

def print_all_uni_analysis_reports(df,columnNames):
    seperator = '---------------------------------------------'
    for column in columnNames:
        analysis = UniVariateAnalysis(df, column)
        analysis_report = UniVariateReport(analysis)

        print(seperator)
        print(f'\'{column}\' column univariate analysis report')
        print(seperator)

        analysis_report.print_report()

# %% [markdown]
# ### 1. UniVariate Analysis

# %%
print_all_uni_analysis_reports(df = concrete_df, columnNames=columns)

# %% [markdown]
# ### 2. Bivariate Analysis
# %% [markdown]
# ### Cement
# Strong relationship, seems to be a mostly positive correlation between this column and the strength of the column. Thouhgh it does oscilate some - but we may be able to attribute the dips to other factors as well (ie maybe the dips had more water in them which weakened it)

# %%
x_column_name = 'cement'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Slag
# There is a very small relationship between slag and target column, the data points are pretty scattered. Thought that maybe there were outliers that were pulling it in different directions but it does not seem that removing the outliers for that column did much. Good candiate for further invistation in conjunction with other columns
# 

# %%
x_column_name = 'slag'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Ash
# Seems to have a slight negative relationship to target even without any outliers. Considering the domain, a little research suggests that there should be a positive relationship between fly ash and the strength of a concrete. This data does not reinforce that idea, which is interesting. Perhaps this column is not the column driving the relationship in a negative direction. This column should be considered with other columns

# %%
x_column_name = 'ash'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Water
# 
# Pretty strong negative relationship between this column and the target variable. Makes sense with respect to the domain we are considering, common sense would suggest that the more water you add, you dilute the mixture which could cause weak concrete.

# %%
x_column_name = 'water'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Superplastic
# Strong positive relationship here, pretty easy to tell that it is very possible that there is a correlation between plastic and the target column.

# %%
x_column_name = 'superplastic'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Coarseagg
# Although this column has a best fit line that would suggest a negative relationship, I am unconvinced - the graph is oscilatting quite a bit so that may be what is throwing it off. I would be surprised if this had a large affect on the strength of concrete.

# %%
x_column_name = 'coarseagg'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### Fineagg
# This is very similar to the column above, best fit line seems to show a negative relationship. However, the graph oscillates quite a bit here too, like the above column there is not really a steady decrease in the peaks or the valleys of the graph

# %%
x_column_name = 'fineagg'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')


# %%


# %% [markdown]
# ### Age
# This one is interesting, it suggests the posibility of a strong positive relationship between this column and target. When removing the outliers, we can also see that the relationship remains. Conceptually, when thinking in terms of the domain, it makes sense that the concrete would have a max life where beyond that, the frequency of older aged concrete is less likely. This columns woudl be a very good candidate when considering which columns affect the target

# %%
x_column_name = 'age'
analysis = UniVariateAnalysis(concrete_df, x_column_name)
df_no_outlier = analysis.get_df_without_outliers_on_column()


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=concrete_df) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=concrete_df, kind='reg')


# %%
sns.lineplot(x=x_column_name, y=targetColumn, data=df_no_outlier) 


# %%
sns.jointplot(x=x_column_name, y=targetColumn, data=df_no_outlier, kind='reg')

# %% [markdown]
# ### 3. Feature Engineering Techniques
# %% [markdown]
# ### Context
# 
# After looking at the dataset, i noticed that were quite a few zeros in many columns in many rows. So to understand the domain (or context), I researched whether or not the zero values were considered valid. fortunately it does seem like the zeros that occur in ['slag','ash','superplastic'] were valid. I saw many websites claim that this is normal. In a production environment, I would delegate this research to a product person that should be more familiar with the domain and could tell me if the values I am seeing are valid.

# %%
concrete_df.head()

