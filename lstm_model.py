from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, Dropout
import pandas as pd

# def generate_model(shape):
#     model = Sequential()
#     model.add(LSTM(
#         units=50,
#         return_sequences=True,
#         input_shape=(shape[1], shape[-1])
#     ))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=15, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=15, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=15, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#     return model

def generate_model(shape):
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=5,
        # return_sequences=False,
        return_sequences=True,
        input_shape=(shape[1], shape[-1])
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=5, return_sequences=False))
    # model.add(LSTM(units=3, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=10, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model

def make_predictions(model, data):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
    checkpoint = ModelCheckpoint(
        'resources/model/best_model_checkpoint.h5',
        # monitor='val_accuracy',
        # mode='max',
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
    log_dir = "logs/fit"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        data["x_train"], data["y_train"], batch_size=100, epochs=100,
        validation_data=(data["x_test"], data["y_test"]),
        callbacks=[checkpoint, tensorboard_callback, es]
    )
    model.save('resources/model/my_model_sequential.h5')

    hist_df = pd.DataFrame(history.history)
    with open('resources/model/history_sequential.csv', mode='w') as f:
        hist_df.to_csv(f)
