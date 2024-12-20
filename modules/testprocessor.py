import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TestDatasetProcessor:
    def __init__(self, dataframe, reference_columns, training_label_encoders, training_scaler):
        self.df = dataframe.copy()
        self.reference_columns = reference_columns
        self.label_encoders = training_label_encoders  # Encoders from training
        self.scaler = training_scaler  # Scaler from training
        self.processed_X = None

    def handle_problematic_values(self):
        if 'funder' in self.df.columns:
            self.df['funder'] = self.df['funder'].replace('RURAL WATER SUPPLY AND SANITAT', 'OTHER')

    def balanced_encoding(self):
        print("\n### Encoding Categorical and Boolean Columns ###")
        bool_columns = self.df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            self.df[col] = self.df[col].astype(int)

        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                self.df[col] = self.df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
        self.df = self.df.reindex(columns=self.reference_columns, fill_value=0)

        print("Balanced encoding completed successfully.")

    def scale_numeric_features(self):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_columns] = self.scaler.transform(self.df[numeric_columns])

    def process_data(self):
        self.handle_problematic_values()
        self.balanced_encoding()
        self.scale_numeric_features()
        self.processed_X = self.df
        print("Test dataset processing completed successfully.")
