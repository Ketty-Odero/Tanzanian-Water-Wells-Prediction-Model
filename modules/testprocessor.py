import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TestDatasetProcessor:
    def __init__(self, dataframe, reference_columns):
        """
        Initialize the TestDatasetProcessor with a dataframe and reference column structure.

        Parameters:
        dataframe (pd.DataFrame): Input dataframe (new test dataset).
        reference_columns (list): List of reference columns from training data.
        """
        self.df = dataframe
        self.reference_columns = reference_columns
        self.processed_X = None
        self.scaler = StandardScaler()

    def handle_problematic_values(self):
        """
        Replace problematic values with a default category.
        """
        if 'funder' in self.df.columns:
            self.df['funder'] = self.df['funder'].replace('RURAL WATER SUPPLY AND SANITAT', 'OTHER')

    def balanced_encoding(self):
        """
        Perform balanced encoding for categorical and boolean columns.
        """
        print("\n### Encoding Categorical and Boolean Columns ###")

        # Handle boolean columns
        bool_columns = self.df.select_dtypes(include=['bool']).columns
        print(f"Boolean columns: {list(bool_columns)}")
        for col in bool_columns:
            self.df[col] = self.df[col].astype(int)

        # Identify categorical columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_columns)}")

        # Label Encoding for categorical columns
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))

        # One-Hot Encoding for categorical columns
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)

        # Align columns with reference training set
        self.df = self.df.reindex(columns=self.reference_columns, fill_value=0)
        print("Balanced encoding completed successfully.")

    def scale_numeric_features(self):
        """
        Scale numeric features using StandardScaler.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])

    def process_data(self):
        """
        Run all preprocessing steps for the test dataset.
        """
        self.handle_problematic_values()
        self.balanced_encoding()
        self.scale_numeric_features()
        self.processed_X = self.df
        print("Test dataset processing completed successfully.")
