from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LSTM, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
order_reviews_path = '/Users/sviat/Downloads/OneDrive_1_30-04-2024/S_Data/order_reviews.csv'

train_data = pd.read_csv(order_reviews_path)
train_data = train_data[['review_comment_message', 'review_score']]
train_data = train_data.dropna()
train_data = train_data.rename(
    columns={'review_comment_message': 'text', 'review_score': 'score'})
# Train Data Labels
train_data["score"] = train_data["score"].astype('category')
train_data["score_label"] = train_data["score"].cat.codes
train_features, train_labels = train_data['text'], tf.one_hot(
    train_data["score_label"], 5)
d = dict(enumerate(train_data["score"].cat.categories))
print(d)


vocab_size = 15000
vector_size = 100
max_seq_len = 20
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(train_data['text'])
sequences_train = tokenizer.texts_to_sequences(train_data['text'])

padded_train = pad_sequences(
    sequences_train, padding='post', maxlen=max_seq_len)


def lstm_model():
    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_size,
                  output_dim=vector_size,
                  input_length=max_seq_len))
    model.add(Dropout(0.6))
    model.add(Bidirectional(LSTM(max_seq_len, return_sequences=True)))
    model.add(Bidirectional(LSTM(5)))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  patience=4,
                                  verbose=1,
                                  mode="min",
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath='models/lstm.keras',
                                    verbose=1,
                                    save_best_only=True)
]
model = lstm_model()
model.summary()

tf.config.run_functions_eagerly(True)
history = model.fit(padded_train,
                    train_labels,
                    validation_split=0.2,
                    callbacks=callbacks,
                    epochs=20)
