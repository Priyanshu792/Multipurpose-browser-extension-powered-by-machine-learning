'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import kagglehub

# Download Kaggle dataset
path = kagglehub.dataset_download("venky73/spam-mails-dataset")
spam_df = pd.read_csv(f"{path[0]}")
print(path)




"""
# Original dataset loading code
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
spam_df = None

for encoding in encodings:
    try:
        spam_df = pd.read_csv('spam_emails.csv', 
                             encoding=encoding,
                             quoting=1,  # Quote all fields
                             quotechar='"',  # Use double quote as quote character
                             delimiter='\t',  # Use tab as delimiter
                             on_bad_lines='skip')  # Skip problematic lines
        print(f"Successfully loaded dataset with {encoding} encoding!")
        break
    except UnicodeDecodeError:
        print(f"Failed to load with {encoding} encoding, trying next...")
        continue

if spam_df is None:
    raise Exception("Failed to load the dataset with any of the attempted encodings")
"""

print(f"Number of spam examples: {len(spam_df)}")
print("\nDataset columns:", spam_df.columns.tolist())
print("\nFirst few rows:")
print(spam_df.head())

# Create combined dataset
combined_data = {
    'Text': [],
    'Category': []
}

# Add spam data (from Kaggle dataset)
for _, row in spam_df.iterrows():
    try:
        # Adjust column names based on Kaggle dataset structure
        text = f"{row['subject']} {row['message']}"  # Combine subject and message
        combined_data['Text'].append(text)
        combined_data['Category'].append(1)  # 1 for spam
    except Exception as e:
        print(f"Error processing row: {e}")
        continue

print(f"\nProcessed {len(combined_data['Text'])} spam examples")

# Add ham examples
ham_texts = [
    "meeting tomorrow at 2pm to discuss project updates",
    "please review the attached document and provide feedback",
    "reminder: team lunch scheduled for friday",
    "quarterly report is ready for review",
    "new office policy regarding work from home",
    "birthday celebration in conference room at 3pm",
    "system maintenance scheduled for weekend",
    "please submit your expenses by end of month",
    "training session on new software tomorrow",
    "vacation request approved for next week",
    
    # Add security notification examples
    "Security alert: New sign-in to your account from Windows device. If this was you, no action needed.",
    "Your Google Account: We detected a new sign-in from Chrome on Windows.",
    "Account security notification: New browser sign-in detected.",
    "Important security alert: Verify recent account activity from new device.",
    "Gmail security: New sign-in from Windows PC. Check activity at https://myaccount.google.com/notifications",
    "Account activity alert: Sign-in detected from new Windows device in your area.",
    "Security notice: Recent sign-in from new device. Review activity in your Google Account.",
    "Google security alert: Confirm recent sign-in activity on Windows.",
    "Account notification: New device added to your trusted devices list.",
    "Security update: Review recent account activity and sign-ins."
]

for text in ham_texts:
    combined_data['Text'].append(text)
    combined_data['Category'].append(0)  # 0 for ham

print(f"\nAdded {len(ham_texts)} ham examples")

# Create final DataFrame
final_df = pd.DataFrame(combined_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    final_df['Text'],
    final_df['Category'],
    test_size=0.2,
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create and train the pipeline with enhanced text processing
clf = Pipeline([
    ('vectorizer', CountVectorizer(
        stop_words='english',
        min_df=1,  # Lower minimum document frequency
        ngram_range=(1, 5),  # Increase phrase length
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',  # Better token pattern
        max_features=10000  # Increase vocabulary size
    )),
    ('nb', MultinomialNB(alpha=0.05))  # Reduce smoothing for more decisive predictions
])

# Train the model
clf.fit(X_train, y_train)

# Evaluate
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"\nTraining accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("\nModel saved as 'model.pkl'")

# Test a few examples
print("\nTesting examples:")
test_texts = [
    "free money win now lottery",
    "meeting tomorrow at 2pm",
    "claim your prize now limited time",
    "project status update report"
]

for text in test_texts:
    prediction = clf.predict([text])[0]
    probability = clf.predict_proba([text])[0]
    confidence = probability.max() * 100
    result = "Spam" if prediction == 1 else "Ham"
    print(f"\nText: {text}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
'
'''


# %% Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline
import pickle

# %% Download necessary NLTK datasets
nltk.download('stopwords')

# %% Load Dataset
df = pd.read_csv("spam_ham_dataset.csv", encoding='ISO-8859-1')

# %% Check data distribution
print(df['label'].value_counts())

# %% Visualize spam vs ham distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=df["label"].value_counts().index, y=df["label"].value_counts().values, palette="viridis")
plt.title("Distribution of Spam vs Ham Emails", fontsize=14)
plt.xlabel("Email Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# %% Pie chart for percentage distribution
plt.figure(figsize=(6, 6))
plt.pie(df["label"].value_counts(normalize=True) * 100, 
        labels=df["label"].value_counts().index, autopct='%1.1f%%', 
        colors=["skyblue", "orange"], startangle=55, wedgeprops={'edgecolor': 'black'})
plt.title("Spam vs Ham Email Distribution", fontsize=14)
plt.show()

# %% Encode labels before dropping
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# %% Drop unnecessary columns
df.drop(['Unnamed: 0', 'label'], axis=1, inplace=True)

# %% Initialize text preprocessing tools
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Define text preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]  # Remove stopwords & stem
    return ' '.join(words)

# %% Apply preprocessing to the dataset
df["clean_text"] = df["text"].apply(preprocess)

# %% Define features and labels
X = df['clean_text']
y = df['label_num']

# %% Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %% Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# %% Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# %% Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %% Classification report
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# %% Function to predict spam messages
def predict_spam(text):
    text = preprocess(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

# %% Sample test cases
test_texts = [
    "Congratulations! You've been selected as a winner. TEXT WON to 44255 to claim your prize.",
    "Reminder: You have an appointment with the doctor tomorrow at 3 PM.",
    "Your Amazon order has been dispatched. Your order ID is #251-1981528-982.",
    "Alert: Your account has been compromised. Please reset your password immediately.",
    "Congratulations! You've won a free trip to Hawaii. Text 'WON' to 555-1234 to claim your reward.",
    "You have a new message from your bank. Please check your account for details.",
    "Your package has been delivered. Click here to track your order.",
    "You've received a new notification from your social media account. Check it out!",
]

# %% Predict on test messages
for text in test_texts:
    result = predict_spam(text)
    print(f"Message: {text}")
    print(f"Prediction: {result}")

# %% Create a pipeline for better integration
spam_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TF-IDF Vectorizer
    ('classifier', MultinomialNB())     # Naive Bayes Classifier
])

# Train pipeline on full dataset
spam_pipeline.fit(X, y)

# %% Save the trained pipeline
with open('spam_pipeline.pkl', 'wb') as file:
    pickle.dump(spam_pipeline, file)

# %%
