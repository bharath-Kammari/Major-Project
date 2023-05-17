# Import necessary libraries
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('task_humanitarian_text_img_train.csv')

# Preprocess the text data
max_features = 2000 # the maximum number of words to keep in the vocabulary
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['tweet_text'].values)
X = tokenizer.texts_to_sequences(df['tweet_text'].values)
X = pad_sequences(X) # pads sequences to the same length
Y = pd.get_dummies(df['label_text']).values

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the LSTM model architecture
embedding_size = 128 # the size of the word embedding layer
lstm_units = 128 # the number of units in the LSTM layer
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=X.shape[1]))
model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the LSTM model
batch_size = 32 # the number of samples per gradient update
epochs = 10 #the number of epochs to train the model
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the LSTM model
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])

# y_pred = model.predict([X_test, Y_test])
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(Y_test, axis=1)
# acc = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# print("Accuracy: {:.2f}%".format(acc * 100))
# print("Precision: {:.2f}%".format(precision * 100))
# print("F1 Score: {:.2f}%".format(f1 * 100))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(Y_test, axis=1)

print(classification_report(y_test, y_pred))

