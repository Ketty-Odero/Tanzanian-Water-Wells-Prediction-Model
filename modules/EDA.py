import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


# Define the EDA class
class EDA: 
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, low_memory=False)  # Optimize memory usage for large files
        self.target_variable = "status_group"

    def univariate_analysis(self):
        print("\n\n======= Univariate Analysis =======\n")

        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = ['amount_tsh', 'gps_height']

        # Categorical Features
        print("\n----- Categorical Features -----\n")
        for col in categorical_cols:
            if self.df[col].nunique() <= 25:  # Limit to max 25 unique values for faster plotting
                if self.df[col].nunique() <= 3:
                    plt.figure(figsize=(6, 6))
                    self.df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                else:
                    plt.figure(figsize=(10, 6))
                    self.df[col].value_counts().plot(kind='bar')
                
                plt.title(f'Distribution of {col}', fontsize=14)
                plt.ylabel('Count' if self.df[col].nunique() > 3 else '')
                plt.xticks(rotation=45)  # Rotate labels for better readability
                plt.show()
            else:
                print(f"Skipping {col} (too many unique values: {self.df[col].nunique()})")
                
        # Numerical Features
        print("\n----- Numerical Features (amount_tsh, gps_height) -----\n")
        for col in numerical_cols:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], bins=20)  # Use fewer bins for faster rendering
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.subplot(1, 2, 2)
            sns.boxplot(y=self.df[col])
            plt.title(f'Boxplot of {col}', fontsize=14)
            plt.tight_layout()
            plt.show()
        
    def bivariate_analysis(self):
        """Performs bivariate analysis with a correlation matrix heatmap."""
        print("\n\n======= Bivariate Analysis (Correlation Matrix Heatmap) =======\n")

        # Select numerical and low-cardinality categorical columns
        df_corr = self.df.select_dtypes(include=['number'])
        for col in self.df.select_dtypes(include=['object']):
            if self.df[col].nunique() <= 25:
                df_corr[col] = self.df[col].astype('category').cat.codes

        # Correlation matrix and heatmap
        corr_matrix = df_corr.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap", fontsize=16)
        plt.show()    

    def multivariate_analysis(self):
        """Performs multivariate analysis with a heatmap based on latitude, longitude and status_group."""
        print("\n\n======= Multivariate Analysis (Geographical Distribution by Status) =======\n")

        plt.figure(figsize=(12, 8))

        # Define a custom colormap for the status_group
        status_colors = {"functional": "green", "non functional": "red", "functional needs repair": "orange"}

        # Normalize the case of the target variable values
        self.df[self.target_variable] = self.df[self.target_variable].str.lower()

        # Create scatterplot and color based on the status
        plt.scatter(
            self.df['longitude'], 
            self.df['latitude'], 
            c=self.df[self.target_variable].map(status_colors), 
            alpha=0.5
        )

        plt.title("Geographical Distribution by Well Status", fontsize=16)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.colorbar(label="Well Status")
        plt.show()

