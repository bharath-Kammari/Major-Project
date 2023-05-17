# Import necessary libraries
import pandas as pd
import torch
import transformers as tf
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('task_informative_text_img_train.csv')

# Preprocess the text data
tokenizer = tf.XLNetTokenizer.from_pretrained('xlnet-base-cased')
encoded_inputs = tokenizer(df['tweet_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(['informative', 'not_informative'])

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_inputs, labels, test_size=0.2, random_state=42)

# Fine-tune the XLNet model
model = tf.XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
optimizer = tf.AdamW(model.parameters(), lr=1e-5)
train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(5):
    for batch in train_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluate the model
model.eval()
test_inputs = {key: val.to(device) for key, val in test_inputs.items()}
with torch.no_grad():
    outputs = model(**test_inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
accuracy = (predictions == test_labels).float().mean()
print('Accuracy:', accuracy.item())
