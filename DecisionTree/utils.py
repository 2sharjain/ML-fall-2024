from collections import Counter

def preprocessing_bank_data(bank_df, numerical_thresholds, numerical_columns):

    for column in numerical_columns:
        bank_df[column] = bank_df[column].apply(lambda x: 0 if x <= numerical_thresholds[column] else 1)

    return bank_df


def replace_unknown_data(bank_df, categorical_columns_with_unknown_values):

    for column in categorical_columns_with_unknown_values:
        values_with_frequency = Counter(bank_df[column]).most_common(2)

        mode = [value for value, frequency in values_with_frequency if value != 'unknown'][0]

        bank_df[column] = bank_df[column].apply(lambda x: mode if x == 'unknown' else x)

    return bank_df