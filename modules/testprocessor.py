import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TestDatasetProcessor:
    def __init__(self, dataframe, reference_X_train, label_encoders, scaler):
        """
        Initialize the TestDatasetProcessor with a dataframe and reference processing artifacts.

        Parameters:
        dataframe (pd.DataFrame): Input dataframe (new test dataset).
        reference_X_train (pd.DataFrame): The reference training set to align columns.
        label_encoders (dict): LabelEncoders used during training.
        scaler (StandardScaler): Scaler fitted on the training dataset.
        """
        self.df = dataframe
        self.reference_X_train = reference_X_train
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.processed_X = None

    def handle_problematic_values(self):
        """
        Replace problematic values with a default category.
        """
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

        # Encode categorical columns using reference encoders
        for col in categorical_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                self.df[col] = self.df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # One-Hot Encoding for low-cardinality columns
        low_cardinality_cols = [col for col in categorical_columns if col not in self.label_encoders]
        self.df = pd.get_dummies(self.df, columns=low_cardinality_cols, drop_first=True)

        # Align columns with reference training set
        self.df = self.df.reindex(columns=self.reference_X_train.columns, fill_value=0)
        print("Balanced encoding completed successfully.")

    def scale_numeric_features(self):
        """
        Scale numeric features using the provided StandardScaler.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_columns] = self.scaler.transform(self.df[numeric_columns])

    def process_data(self):
        """
        Run all preprocessing steps for the test dataset.
        """
        self.handle_problematic_values()
        self.balanced_encoding()
        self.scale_numeric_features()
        self.processed_X = self.df
        print("Test dataset processing completed successfully.")


