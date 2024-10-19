def preprocess_bank_data(bank_df, thresholds, numerical_columns):

    for column in numerical_columns:
        bank_df[column] = bank_df[column].apply(lambda x: 0 if x <= thresholds[column] else 1)

    return bank_df


