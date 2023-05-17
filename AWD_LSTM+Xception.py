import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, concatenate, Embedding, Bidirectional, GlobalMaxPooling2D, Dropout
from keras.models import Model
from keras.utils import img_to_array, load_img
from keras.utils import pad_sequences
from keras.utils import to_categorical
from transformers import AutoTokenizer

# Load text data
train_data = pd.read_csv('task_informative_text_img_train.csv')
test_data = pd.read_csv('task_informative_text_img_test.csv')
val_data = pd.read_csv('task_informative_text_img_dev.csv')

X1_text=[]
Y1_text=[]
X2_text=[]
Y2_text=[]
X3_text=[]
Y3_text=[]
# train
for i in range(0,13608):
    if train_data['label_text'][i]==train_data['label_image'][i]:
        X1_text.append(train_data['tweet_text'][i])
        Y1_text.append(train_data['label_text'][i])

for i in range(0,2237):
    if test_data['label_text'][i]==test_data['label_image'][i]:
        X2_text.append(test_data['tweet_text'][i])
        Y2_text.append(test_data['label_text'][i])

for i in range(0,2237):
    if val_data['label_text'][i]==val_data['label_image'][i]:
        X3_text.append(val_data['tweet_text'][i])
        Y3_text.append(val_data['label_text'][i])
# Tokenize text data
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X1_text)
sequences_train = tokenizer.texts_to_sequences(X1_text)
sequences_test = tokenizer.texts_to_sequences(X2_text)
sequences_val = tokenizer.texts_to_sequences(X3_text)
x_train = pad_sequences(sequences_train, maxlen=max_len)
x_test = pad_sequences(sequences_test, maxlen=max_len)
x_val = pad_sequences(sequences_val, maxlen=max_len)

X1_image=[]
Y1_image=[]
X2_image=[]
Y2_image=[]
X3_image=[]
Y3_image=[]
# train
for i in range(0,13608):
    if train_data['label_text'][i]==train_data['label_image'][i]:
        X1_image.append(train_data['image'][i])
        Y1_image.append(train_data['label_image'][i])

for i in range(0,2237):
    if test_data['label_text'][i]==test_data['label_image'][i]:
        X2_image.append(test_data['image'][i])
        Y2_image.append(test_data['label_image'][i])

for i in range(0,2237):
    if val_data['label_text'][i]==val_data['label_image'][i]:
        X3_image.append(val_data['image'][i])
        Y3_image.append(val_data['label_image'][i])
# Load image data
img_width, img_height = 100, 100
num_classes = 2
img_train = []
img_test = []
img_val = []


for filename in X1_image:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_train.append(img_array)

for filename in X2_image:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_test.append(img_array)

for filename in X3_image:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_val.append(img_array)

img_train = np.array(img_train)
img_test = np.array(img_test)
img_val = np.array(img_val)

# Define LSTM model
max_len = 100
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
awd_lstm_input = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(tokenizer.vocab_size, 128)(awd_lstm_input)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
global_max_pooling = GlobalMaxPooling2D()(lstm_layer)
dropout_layer = Dropout(0.2)(global_max_pooling)
awd_lstm_output = Dense(1, activation='sigmoid')(dropout_layer)
awd_lstm_model = Model(inputs=awd_lstm_input, outputs=awd_lstm_output)

# Define Xception model
xception_input = Input(shape=(img_width, img_height, 3))
xception_model = Xception(weights='imagenet', include_top=False, input_tensor=xception_input, pooling='max')
xception_output = Dense(num_classes, activation='sigmoid')(xception_model.output)
xception_model = Model(inputs=xception_input, outputs=xception_output)

# Combine LSTM and Xception models with intermediate fusion
print(np.shape(lstm_model.output))
print(np.shape(xception_model.output))
combined_input = concatenate([lstm_model.output, xception_model.output])
fusion_output = Dense(num_classes, activation='sigmoid')(combined_input)
fusion_model = Model(inputs=[lstm_model.input, xception_model.input], outputs=fusion_output)

# Compile the fusion model
fusion_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the fusion model
tempTr=[]
tempTe=[]
tempDe=[]

for i in range(0, len(Y1_text)):
    if Y1_text[i] == 'informative':
        tempTr.append(0)
    else:
        tempTr.append(1)
for i in range(0, len(Y2_text)):
    if Y2_text[i] == 'informative':
        tempTe.append(0)
    else:
        tempTe.append(1)
for i in range(0, len(Y3_text)):
    if Y3_text[i] == 'informative':
        tempDe.append(0)
    else:
        tempDe.append(1)
labels_train = tempTr
labels_test = tempTe
labels_val = tempDe
y_train = to_categorical(labels_train, num_classes=num_classes)
y_test = to_categorical(labels_test, num_classes=num_classes)
y_val = to_categorical(labels_val, num_classes=num_classes)

history = fusion_model.fit([x_train, img_train], y_train,
                           epochs=15, batch_size=40,
                           validation_data=([x_val, img_val], y_val))

# Evaluate the fusion model
score = fusion_model.evaluate([x_test, img_test], y_test, verbose=0)

# Print accuracy, precision, and F1 score
y_pred = fusion_model.predict([x_test, img_test])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy: {:.2f}%".format(acc*100))
print("Precision: {:.2f}%".format(precision*100))
print("F1 Score: {:.2f}%".format(f1*100))



