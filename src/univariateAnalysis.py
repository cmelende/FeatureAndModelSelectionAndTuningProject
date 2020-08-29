
class UniVariateAnalysis:

    def __init__(self, df, columnName):
        self.columnName = columnName
        self.dataframe = df
        self.series = df[columnName]
    
    def get_q1(self):
        return self.series.quantile(.25)

    def get_q2(self):
        return self.series.quantile(.5)

    def get_q3(self):
        return self.series.quantile(.75)

    def get_q4(self):
        return self.series.quantile(1)

    def get_iqr(self):
        return self.get_q3() - self.get_q1()

    def get_min(self):
        return self.dataframe[self.columnName].min()

    def get_median(self):
        return self.dataframe[self.columnName].median()

    def get_max(self):
        return self.dataframe[self.columnName].max()

    def get_data_type(self):
        return self.dataframe[self.columnName].dtypes

    def get_lower_outlier_rows(self):
        return self.dataframe.loc[(self.dataframe[self.columnName] < self.get_lower_whisker_value())]

    def get_lower_whisker_value(self):
        return self.get_q1() - ((3/2) * self.get_iqr())

    def get_higher_outlier_rows(self):
        return self.dataframe.loc[(self.dataframe[self.columnName] > self.get_higher_whisker_value())]

    # def get_higher_outlier_indices(self):
    #     return self.dataframe[self.dataframe[self.columnName] > self.get_higher_whisker_value()].index.values.astype(int)

    def get_higher_whisker_value(self):
        return self.get_q3() + ( (3/2) * self.get_iqr())

    def get_std(self):
        return self.series.std()

    def get_mean(self):
        return self.series.mean()


    def get_df_without_lower_outliers_on_column(self):
        copy = self.dataframe.copy()
        copy = copy[copy[self.columnName] >= self.get_lower_whisker_value()]
        return copy

    def get_df_without_higher_outliers_on_column(self):
        copy_df = self.df.copy()
        copy_df = copy_df[copy_df[self.columnName] <= self.get_higher_whisker_value()]
        return copy_df

    def get_df_without_outliers_on_column(self):
        copy_df = self.dataframe.copy()
        copy_df = copy_df[(copy_df[self.columnName] <= self.get_higher_whisker_value()) & (copy_df[self.columnName] >= self.get_lower_whisker_value())]
        return copy_df

    
class OutlierFilter:
    def __init__(self, df, filterColumns):
        self.dataframe = df
        self.columnNames = filterColumns
        
    # def get_lower_whisker_value(self , columnName):
    #     analysis = UniVariateAnalysis(self.dataframe, columnName)
    #     return analysis.get_q1() - ((3/2) * analysis.get_iqr())

    # def get_higher_whisker_value(self, columnName):
    #     analysis = UniVariateAnalysis(self.dataframe, columnName)
    #     return analysis.get_q3() + ( (3/2) * analysis.get_iqr())

    
    def get_df_without_lower_outliers(self):
        copy = self.dataframe.copy()
        for col in self.columnNames:
            analysis = UniVariateAnalysis(self.dataframe, col)
            copy = copy[copy[col] >= analysis.get_lower_whisker_value()]
        return copy

    def get_df_without_higher_outliers(self):
        copy_df = self.dataframe.copy()
        for col in self.columnNames:
            analysis = UniVariateAnalysis(self.dataframe, col)
            copy_df = copy_df[copy_df[col] <= analysis.get_higher_whisker_value()]
        return copy_df

    def get_df_without_outliers(self):
        copy_df = self.dataframe.copy()
        for col in self.columnNames:
            analysis = UniVariateAnalysis(self.dataframe, col)
            copy_df = copy_df[(copy_df[col] <= analysis.get_higher_whisker_value()) & (copy_df[col] >= analysis.get_lower_whisker_value())]
        return copy_df

        
class UniVariateReport: 
    def __init__(self, uniVariateAnalysis):
        self.analysis = uniVariateAnalysis
    
    def print_quartiles(self):
        print("Q1: " , self.analysis.get_q1())
        print("Q2: ", self.analysis.get_q2())
        print("Q3: ", self.analysis.get_q3())
        print("Q4: ", self.analysis.get_q4())
        print("Mean: ", self.analysis.get_mean())
        print("Min: ", self.analysis.get_min())
        print("Median: ", self.analysis.get_median())
        print("Max: ", self.analysis.get_max())
    
    def print_whiskers(self):
        print("Top whisker: ", self.analysis.get_higher_whisker_value())
        print("Bottom whisker: ", self.analysis.get_lower_whisker_value())

    def print_data_type(self):
        print("Data type: ", self.analysis.get_data_type())

    def print_value_range(self):
        print(f'Range of values: ({self.analysis.get_min()}, {self.analysis.get_max()})')

    def print_std(self):
        print("Standard deviation: ", self.analysis.get_std())

    def print_higher_outlier_indices(self):
        indices = self.analysis.get_higher_outlier_rows().index
        print("Number of outliers above the top whisker: ", indices.size)
        counter = 1
        if indices.size > 0:
            print("Indices of higher outlier rows")
            for index in indices:
                print(f'{counter}) {index}')
                counter += 1

    def print_lower_outlier_indices(self):
        indices = self.analysis.get_lower_outlier_rows().index
        print("Number of outliers below the bottom whisker: ", indices.size)
        counter = 1
        if indices.size > 0:
            print("Indices of bottom outlier rows v3")
            for index in indices:
                print(f'{counter})     {index}')
                counter += 1

    def print_report(self):
        self.print_data_type()
        self.print_value_range()
        self.print_std()
        self.print_quartiles()
        self.print_whiskers()
        self.print_higher_outlier_indices()
        self.print_lower_outlier_indices()
