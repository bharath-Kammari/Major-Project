



max_len = 100
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
awd_lstm_input = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(tokenizer.vocab_size, 128)(awd_lstm_input)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
global_max_pooling = GlobalMaxPooling2D()(lstm_layer)
dropout_layer = Dropout(0.2)(global_max_pooling)
awd_lstm_output = Dense(1, activation='sigmoid')(dropout_layer)
awd_lstm_model = Model(inputs=awd_lstm_input, outputs=awd_lstm_output)