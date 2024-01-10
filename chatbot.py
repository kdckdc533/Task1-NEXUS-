# Import necessary libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load intents from a JSON file containing predefined patterns and responses for the chatbot
intents = json.loads(open('intents.json').read())

# Load pickled data: words and classes for the chatbot model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the pre-trained chatbot model
model = load_model('chatbot_model.model')

# Initialize a WordNet Lemmatizer for preprocessing text data
lemmatizer = WordNetLemmatizer()

# Function to clean up a sentence by tokenizing and lemmatizing words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize each word
    return sentence_words

# Function to create a bag of words representation from a sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Initialize a bag with zeros for each word in the vocabulary
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Set 1 at the index corresponding to the word in the bag
    return np.array(bag)  # Return the bag of words as a NumPy array

# Function to predict the class/intent of a given sentence using the trained model
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Get the bag of words for the sentence
    res = model.predict(np.array([bow]))[0]  # Predict the probabilities for each class/intent
    ERROR_THRESHOLD = 0.25  # Threshold to consider a prediction valid
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter results above threshold

    results.sort(key=lambda x: x[1], reverse=True)  # Sort results by probability in descending order
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})  # Return predicted intents
    return return_list

# Function to retrieve a random response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']  # Get the predicted intent tag
    list_of_intents = intents_json['intents']  # Get the list of intents from the loaded JSON data
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Select a random response from the chosen intent
            break
    return result

# Main loop to continuously accept user input and provide responses
print("GO! Bot is running")
while True:
    message = input("")  # Get user input
    ints = predict_class(message)  # Predict the intent of the user input
    res = get_response(ints, intents)  # Get a response based on the predicted intent
    print(res)  # Print the response to the user
