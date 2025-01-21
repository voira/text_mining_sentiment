import numpy as np
import csv
import random
import sklearn


# Load dataset
def load_dataset(file_path):
    reviews, sentiments = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            reviews.append(row[0].lower())
            sentiments.append(1 if row[1].strip().lower() == "positive" else 0)
    return reviews, np.array(sentiments)

# Split dataset
def split_dataset(reviews, sentiments, train_ratio=0.8):
    indices = list(range(len(reviews)))
    random.shuffle(indices)
    split_idx = int(len(reviews) * train_ratio)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_data = [reviews[i] for i in train_indices], sentiments[train_indices]
    val_data = [reviews[i] for i in val_indices], sentiments[val_indices]
    return train_data, val_data

# Build vocabulary
def build_vocab(reviews):
    vocab = {}
    for review in reviews:
        for word in review.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Start indices from 1
    return vocab

# Convert text to vector using TF-IDF
def text_to_vector(reviews, vocab=None, vector_dim=100):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=vocab, max_features=vector_dim)
    vectors = vectorizer.fit_transform(reviews).toarray()
    return vectors

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Neural Network
class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, Y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Metrics
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred >= 0.5))
    fp = np.sum((y_true == 0) & (y_pred >= 0.5))
    fn = np.sum((y_true == 1) & (y_pred < 0.5))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

# Main script
file_path = "C:/Users/beste/Downloads/c.csv"  # Replace with your dataset file path
reviews, sentiments = load_dataset(file_path)
(train_reviews, train_labels), (val_reviews, val_labels) = split_dataset(reviews, sentiments)

vocab = build_vocab(train_reviews)
train_vectors = text_to_vector(train_reviews, vocab)
val_vectors = text_to_vector(val_reviews, vocab)

input_size = train_vectors.shape[1]
hidden_size = 10
output_size = 1

model = FeedForwardNN(input_size, hidden_size, output_size)
learning_rate = 0.01

# Training with batches
batch_size = 32
num_batches = len(train_vectors) // batch_size

for epoch in range(500):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_vectors = train_vectors[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx].reshape(-1, 1)
        
        predictions = model.forward(batch_vectors)
        model.backward(batch_vectors, batch_labels, learning_rate)
    
    if epoch % 50 == 0:
        train_loss = -np.mean(train_labels * np.log(predictions) + (1 - train_labels) * np.log(1 - predictions))
        print(f"Epoch {epoch}, Loss: {train_loss}")

# Evaluation
val_predictions = model.forward(val_vectors).flatten()
precision, recall, f1_score = calculate_metrics(val_labels, val_predictions)
print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")