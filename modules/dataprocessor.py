import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self, dataframe, target_column):
        """
        Initialize the DataProcessor with a dataframe and target column.

        Parameters:
        dataframe (pd.DataFrame): Input dataframe.
        target_column (str): The name of the target column.
        """
        self.df = dataframe
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_encoded = None
        self.X_test_encoded = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataframe into training and testing sets.
        """
        print("\n### Splitting Data ###")
        X = self.df.drop([self.target_column, 'id', 'date_recorded'], axis=1, errors='ignore')
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Training set size: {self.X_train.shape[0]} rows")
        print(f"Testing set size: {self.X_test.shape[0]} rows")

    def handle_problematic_values(self):
        """
        Replace problematic values with a default category.
        """
        self.X_train['funder'] = self.X_train['funder'].replace('RURAL WATER SUPPLY AND SANITAT', 'OTHER')
        self.X_test['funder'] = self.X_test['funder'].replace('RURAL WATER SUPPLY AND SANITAT', 'OTHER')

    def balanced_encoding(self):
        """
        Perform balanced encoding for categorical and boolean columns.
        """
        print("\n### Encoding Categorical and Boolean Columns with Balanced Approach ###")
        
        # Handle boolean columns
        bool_columns = self.X_train.select_dtypes(include=['bool']).columns
        print(f"Boolean columns: {list(bool_columns)}")
        for col in bool_columns:
            self.X_train[col] = self.X_train[col].astype(int)
            self.X_test[col] = self.X_test[col].astype(int)

        # Identify categorical columns
        categorical_columns = self.X_train.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_columns)}")

        # Split into high and low-cardinality columns
        high_cardinality_cols = [col for col in categorical_columns if self.X_train[col].nunique() > 10]
        low_cardinality_cols = [col for col in categorical_columns if self.X_train[col].nunique() <= 10]

        # Label Encoding for high-cardinality columns
        for col in high_cardinality_cols:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
            self.X_test[col] = self.X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            self.label_encoders[col] = le

        # One-Hot Encoding for low-cardinality columns
        self.X_train = pd.get_dummies(self.X_train, columns=low_cardinality_cols, drop_first=True)
        self.X_test = pd.get_dummies(self.X_test, columns=low_cardinality_cols, drop_first=True)

        # Align columns in train and test sets
        self.X_test = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)

        print("Balanced encoding completed successfully.")

    def encode_target(self):
        """
        Encode the target column.
        """
        label_encoder_y = LabelEncoder()
        self.y_train_encoded = pd.Series(
            label_encoder_y.fit_transform(self.y_train),
            index=self.y_train.index,
            name=self.target_column
        )
        self.y_test_encoded = pd.Series(
            label_encoder_y.transform(self.y_test),
            index=self.y_test.index,
            name=self.target_column
        )
        self.label_encoders[self.target_column] = label_encoder_y

        print("Target Encoding Mapping:")
        print(dict(zip(label_encoder_y.classes_, range(len(label_encoder_y.classes_)))))

    def scale_numeric_features(self):
        """
        Scale numeric features using StandardScaler.
        """
        numeric_columns = self.X_train.select_dtypes(include=['float64', 'int64']).columns
        self.X_train[numeric_columns] = self.scaler.fit_transform(self.X_train[numeric_columns])
        self.X_test[numeric_columns] = self.scaler.transform(self.X_test[numeric_columns])

    def check_and_fix_encoding(self):
        """
        Ensure all columns are properly encoded.
        """
        non_numeric_columns = self.X_train.select_dtypes(include=['object']).columns
        for col in non_numeric_columns:
            print(f"Encoding column: {col}")
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
            self.X_test[col] = self.X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    def process_data(self):
        """
        Run all data processing steps.
        """
        self.split_data()
        self.handle_problematic_values()
        self.balanced_encoding()
        self.encode_target()
        self.scale_numeric_features()
        print("Data processing completed successfully.")


