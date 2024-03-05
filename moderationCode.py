import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from csv import writer
#for synonyms words
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')
def code_reload():
    # Load dataset
    data = pd.read_csv('English.csv')
    
    # Check for any missing values
    data.dropna(inplace=True)
    
    # Split data into features (X) and target variable (y)
    X = data['text']
    y = data['label']
    
    # Convert text data into TF-IDF features
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Initialize logistic regression model
    model = LogisticRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    return tfidf_vectorizer,model
    
    # Example of using the model for inference
def predict_word_status(tfidf_vectorizer,model,word):
    tfidf_vectorizer,model=code_reload()
    word_tfidf = tfidf_vectorizer.transform([word])
    prediction = model.predict(word_tfidf)
    # print("prints status:",prediction[0]," ",word) 
    if prediction[0] == 1:
        return "Bad"
    else:
        return "Good"

#Code for add into datasets
def AddData(usersdata):
  List = [usersdata,1]
  # Open our existing CSV file in append mode
  # Create a file object for this file
  with open('English.csv', 'a') as f_object:
	# Pass this file object to csv.writer()
	# and get a writer object
       writer_object=writer(f_object)
       writer_object.writerow(List)
	     # Close the file object
       f_object.close()

#Synonyms code
def find_synonyms(word):
    vector,model=code_reload()
    synonyms = set()
    f=0
    count_bad=0
    count_good=0
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if(predict_word_status(vector,model,lemma.name())=='Bad'):
              count_bad=count_bad+1
              # print(f"The word '{user_input}' is classified as Bad.")
              # print("badWord:",lemma.name())
              f=1
              break
            else:
              count_good=count_good+1
        if(f==1):
          if(count_good>=2):
            return "Good"
          else:
            # synonyms.add(lemma.name())
            AddData(word)
            return "Bad"
    return "Good"
    # return synonyms

def passMessage(msg):
   vector,model=code_reload()
   temp_input=list(msg.split(' '))
   for i in temp_input:
      prediction=predict_word_status(vector,model,i)
      print(prediction)
      if (prediction=='Good'):
         word=i
         synonyms=find_synonyms(word)
      else:
         return False
   return True

# Example usage
# user_input = input("Enter a word: ")
# temp_input=list(user_input.split(' '))
# for i in temp_input:
#   prediction = predict_word_status(i)
#   print(i)
#   if(prediction=='Good'):
#      word = i
#      synonyms = find_synonyms(word)
#      print(f"Synonyms of '{word}': {synonyms}")
#   else:
#     print(f"else The word '{user_input}' is classified as {prediction}.")
#     break
  
# inp=input("Do you think it is vulgr word:")
# if(inp=='yes'):
#   AddData(user_input)