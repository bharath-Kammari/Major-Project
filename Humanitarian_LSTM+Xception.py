import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, concatenate, Embedding
from keras.models import Model
from keras.utils import img_to_array, load_img
from keras.utils import pad_sequences
from keras.utils import to_categorical

# Load text data
train_data = pd.read_csv('task_humanitarian_text_img_train.csv')
test_data = pd.read_csv('task_humanitarian_text_img_test.csv')
val_data = pd.read_csv('task_humanitarian_text_img_dev.csv')

# Tokenize text data
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['tweet_text'])
sequences_train = tokenizer.texts_to_sequences(train_data['tweet_text'])
sequences_test = tokenizer.texts_to_sequences(test_data['tweet_text'])
sequences_val = tokenizer.texts_to_sequences(val_data['tweet_text'])
x_train = pad_sequences(sequences_train, maxlen=max_len)
x_test = pad_sequences(sequences_test, maxlen=max_len)
x_val = pad_sequences(sequences_val, maxlen=max_len)

# Load image data
img_width, img_height = 100, 100
num_classes = 8
img_train = []
img_test = []
img_val = []

for filename in train_data['image']:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_train.append(img_array)

for filename in test_data['image']:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_test.append(img_array)

for filename in val_data['image']:
    img = load_img( filename, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_val.append(img_array)

img_train = np.array(img_train)
img_test = np.array(img_test)
img_val = np.array(img_val)

# Define LSTM model
lstm_input = Input(shape=(max_len,))
embedding_layer = Embedding(max_words, 128)(lstm_input)
lstm_layer = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)
lstm_output = Dense(num_classes, activation='softmax')(lstm_layer)
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)

# Define Xception model
xception_input = Input(shape=(img_width, img_height, 3))
xception_model = Xception(weights='imagenet', include_top=False, input_tensor=xception_input, pooling='max')
xception_output = Dense(num_classes, activation='softmax')(xception_model.output)
xception_model = Model(inputs=xception_input, outputs=xception_output)

# Combine LSTM and Xception models with intermediate fusion
combined_input = concatenate([lstm_model.output, xception_model.output])
fusion_output = Dense(num_classes, activation='softmax')(combined_input)
fusion_model = Model(inputs=[lstm_model.input, xception_model.input], outputs=fusion_output)

# Compile the fusion model
fusion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the fusion model
tempTrain=[]
tempTest=[]
tempVal=[]

for i in range(0, 13608):
    if train_data['label'][i] == 'affected_individuals':
        tempTrain.append(0)
    elif train_data['label'][i] == 'infrastructure_and_utility_damage':
        tempTrain.append(1)
    elif train_data['label'][i] == 'injured_or_dead_people':
        tempTrain.append(2)
    elif train_data['label'][i] == 'missing_or_found_people':
        tempTrain.append(3)
    elif train_data['label'][i] == 'not_humanitarian':
        tempTrain.append(4)
    elif train_data['label'][i] == 'other_relevant_information':
        tempTrain.append(5)
    elif train_data['label'][i] == 'rescue_volunteering_or_donation_effort':
        tempTrain.append(6)
    elif train_data['label'][i] == 'vehicle_damage':
        tempTrain.append(7)

for i in range(0, 2237):
    if test_data['label'][i] == 'affected_individuals':
        tempTest.append(0)
    elif test_data['label'][i] == 'infrastructure_and_utility_damage':
        tempTest.append(1)
    elif test_data['label'][i] == 'injured_or_dead_people':
        tempTest.append(2)
    elif test_data['label'][i] == 'missing_or_found_people':
        tempTest.append(3)
    elif test_data['label'][i] == 'not_humanitarian':
        tempTest.append(4)
    elif test_data['label'][i] == 'other_relevant_information':
        tempTest.append(5)
    elif test_data['label'][i] == 'rescue_volunteering_or_donation_effort':
        tempTest.append(6)
    elif test_data['label'][i] == 'vehicle_damage':
        tempTest.append(7)

for i in range(0, 2237):
    if val_data['label'][i] == 'affected_individuals':
        tempVal.append(0)
    elif val_data['label'][i] == 'infrastructure_and_utility_damage':
        tempVal.append(1)
    elif val_data['label'][i] == 'injured_or_dead_people':
        tempVal.append(2)
    elif val_data['label'][i] == 'missing_or_found_people':
        tempVal.append(3)
    elif val_data['label'][i] == 'not_humanitarian':
        tempVal.append(4)
    elif val_data['label'][i] == 'other_relevant_information':
        tempVal.append(5)
    elif val_data['label'][i] == 'rescue_volunteering_or_donation_effort':
        tempVal.append(6)
    elif val_data['label'][i] == 'vehicle_damage':
        tempVal.append(7)
        
labels_train = tempTrain
labels_test = tempTest
labels_val = tempVal
y_train = to_categorical(labels_train, num_classes=num_classes)
y_test = to_categorical(labels_test, num_classes=num_classes)
y_val = to_categorical(labels_val, num_classes=num_classes)

history = fusion_model.fit([x_train, img_train], y_train,
                           epochs=10, batch_size=32,
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

