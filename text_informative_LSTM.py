import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('task_informative_text_img_train.csv')
X1_text=[]
Y1_text=[]
for i in range(0,13608):
    if df['label_text'][i]==df['label_image'][i]:
        X1_text.append(df['tweet_text'][i])
        Y1_text.append(df['label_text'][i])
# Preprocess the text data
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X1_text)
X = tokenizer.texts_to_sequences(X1_text)
X = pad_sequences(X)
Y = pd.get_dummies(Y1_text)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the LSTM model architecture
embedding_size = 128
lstm_units = 128
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=X.shape[1]))
model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the LSTM model
batch_size = 32
epochs = 20
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the LSTM model
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(Y_test, axis=1)

print(classification_report(y_test, y_pred))
