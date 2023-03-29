import pandas as pd
import random, math

def check_split(df, scaler, CANDLES_HISTORY, X, Y):
    """
        To verify it we have to take the RANDOM_VALUE:CANDLES_HISTORY + RANDOM_VALUE
        position in the Y vector because X is an sliced vector.

        And to get X we have to access the CANDLES_HISTORY + RANDOM_VALUE position,
        because for each position it stores a slice of rows, from 

        Example:
        CANDLES_HISTORY = 3
        RANDOM_VALUE = 0
        Y[0:3] -> [1, 1, 0]
        X[0]: From row 0 to 2
        X[1]: From row 1 to 3
        X[2]: From row 2 to 4

        Notice that in the original data were deleted the amount of rows
        used to compute the indicators due to the .dropna()
    """
    RANDOM_VALUE = random.randint(0, len(Y))
    check_split_df = pd.DataFrame(X[RANDOM_VALUE])
    check_split_df["prediction"] = Y[RANDOM_VALUE:CANDLES_HISTORY + RANDOM_VALUE]
    # Here we have to substract 1 because .loc takes the row given the exact label,
    # and vector takes the index - 1 because is 0-indexed
    original_df = scaler.transform(df.loc[RANDOM_VALUE:(CANDLES_HISTORY + RANDOM_VALUE - 1)].values)
    original_df = pd.DataFrame(original_df)
    return (
        (check_split_df.values == original_df.values).all(),
        f"""
            {RANDOM_VALUE}, \n
            {check_split_df.values}, \n
            {original_df.values}, \n
        """
    )

def check_train_data_shape(X, Y):
    return X.shape[0] == Y.shape[0]

def check_total(y_test, Y, PERCENTAGE_DATA, training_data_len, y_train, shrink):
    def get_len(samples):
        return math.ceil(len(samples) * PERCENTAGE_DATA)

    return (
        (y_test.shape[0] + get_len(Y[training_data_len:]))
        +
        (
            shrink - (
                y_train.shape[0] +
                get_len(Y[:training_data_len])
            )
        )
        +
        (y_train.shape[0] + get_len(Y[:training_data_len]))
    )
