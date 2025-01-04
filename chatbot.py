import nltk
import numpy as np
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tkinter as tk

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

epochs = 200
model.fit(train_x, train_y, epochs=epochs, batch_size=5, verbose=1)

model.save('chatbot_model.h5')

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

try:
    model = load_model('chatbot_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        if s in words:
            bag[words.index(s)] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    p = np.reshape(p, (1, len(p)))
    prediction = model.predict(p)
    ERROR_THRESHOLD = 0.7
    
    results = [(i, prob) for i, prob in enumerate(prediction[0]) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if results:
        predicted_index = results[0][0]
        predicted_tag = classes[predicted_index]
        probability = results[0][1]
        return predicted_tag, probability
    else:
        return None, None

def get_response(predicted_tag, intents_json):
    for intent in intents_json['intents']:
        if predicted_tag == intent['tag']:
            return random.choice(intent['responses'])
    return "Sorry, I didn't get a response for that."

def chatbot_response():
    user_input = entry.get()
    if user_input.strip():
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + user_input + "\n")
        
        predicted_tag, probability = predict_class(user_input, model)
        response = get_response(predicted_tag, intents)
        
        chat_history.insert(tk.END, "Chatbot: " + response + "\n\n")
        
        chat_history.yview(tk.END)
        entry.delete(0, tk.END)

root = tk.Tk()
root.title("Chatbot")

chat_history = tk.Text(root, height=20, width=50, wrap=tk.WORD)
chat_history.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

button = tk.Button(root, text="Ask", width=10, command=chatbot_response)
button.pack(pady=5)

root.mainloop()
