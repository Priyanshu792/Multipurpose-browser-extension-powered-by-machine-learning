chrome.runtime.onInstalled.addListener(() => {
    console.log("Email Spam Detector extension installed.");
});

chrome.action.onClicked.addListener((tab) => {
    chrome.identity.getAuthToken({ interactive: true }, (token) => {
        if (chrome.runtime.lastError || !token) {
            console.error(chrome.runtime.lastError);
            return;
        }

        // Make API request to your Flask app with the token
        fetch("http://localhost:5000/classify_emails", {
            method: "POST",
            headers: {
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ token: token })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
        })
        .catch(error => console.error("Error:", error));
    });
});

async function analyzeText(text) {
    console.group('Content Analysis');
    console.log('Analyzing text:', text.substring(0, 150) + '...');
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        const result = await response.json();
        console.log('Analysis result:', result);
        console.groupEnd();
        return result;
    } catch (error) {
        console.error('Analysis error:', error);
        console.groupEnd();
        throw error;
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeContent") {
        analyzeText(request.content)
            .then(result => sendResponse(result))
            .catch(error => sendResponse({ error: error.message }));
        return true; // Required for async response
    }
});
