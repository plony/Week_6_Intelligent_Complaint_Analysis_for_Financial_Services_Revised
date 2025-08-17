import pandas as pd


def load_complaints_data(file_path):
    """
    Loads the CFPB complaints dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
