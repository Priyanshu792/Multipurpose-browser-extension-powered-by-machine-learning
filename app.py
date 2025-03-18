from flask import Flask, request, redirect, url_for, session, render_template
import pickle
import os
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import jsonify

app = Flask(__name__)
app.secret_key = 'YOUR_SECRET_KEY'

# OAuth configuration
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # For testing only; use HTTPS in production
CLIENT_SECRETS_FILE = "client_secret.json"  # Your client secret file
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

@app.route('/')
def home():
    if 'credentials' not in session:
        return redirect('authorize')
    return render_template('index.html')

@app.route('/authorize')
def authorize():
    # Start the OAuth flow
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    # Handle the callback from Google's OAuth server
    state = session['state']
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    # Store the credentials in the session
    session['credentials'] = flow.credentials_to_dict()
    return redirect('/')

@app.route('/list_emails')
def list_emails():
    # Use the stored credentials to access the Gmail API
    if 'credentials' not in session:
        return redirect('authorize')

    credentials = session['credentials']
    service = build('gmail', 'v1', credentials=credentials)

    # Get a list of email messages
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
    messages = results.get('messages', [])

    # Fetch and classify the email content
    email_results = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        email_body = msg['snippet']
        prediction = clf.predict([email_body])[0]
        result = "Spam" if prediction == 1 else "Ham"
        email_results.append({'snippet': email_body, 'result': result})

    return render_template('index.html', emails=email_results)

@app.route('/classify_emails', methods=['POST'])
def classify_emails():
    # Extract the OAuth token from the request
    token = request.json.get('token')

    # Use the token to access the Gmail API
    credentials = google.oauth2.credentials.Credentials(token)
    service = build('gmail', 'v1', credentials=credentials)

    # Get a list of email messages from the inbox
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
    messages = results.get('messages', [])

    email_results = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        email_body = msg['snippet']
        prediction = clf.predict([email_body])[0]
        result = "Spam" if prediction == 1 else "Ham"
        email_results.append({'snippet': email_body, 'result': result})

    return jsonify(email_results)


if __name__ == '__main__':
    app.run(debug=True)
