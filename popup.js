document.addEventListener('DOMContentLoaded', async () => {
    const loadingDiv = document.getElementById('loading');
    const emailsDiv = document.getElementById('emails');
    const summaryDiv = document.getElementById('summary');
    
    loadingDiv.style.display = 'block';
    emailsDiv.innerHTML = '';
    summaryDiv.innerHTML = '';

    try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (!tabs[0]?.id) {
            throw new Error('No active tab found');
        }

        chrome.tabs.sendMessage(tabs[0].id, { action: "getPageContent" }, async (response) => {
            console.log('Received response:', response);
            loadingDiv.style.display = 'none';
            
            if (response?.content?.fullText) {
                // Display complete content
                emailsDiv.innerHTML = `
                    <div class="email">
                        <div class="raw-content">
                            <p><strong>Complete Page Content:</strong></p>
                            <p class="text-preview">${response.content.fullText}</p>
                            <div class="content-stats">
                                <p><strong>Statistics:</strong></p>
                                <p>Characters: ${response.content.stats.length}</p>
                                <p>Words: ${response.content.stats.wordCount}</p>
                                <p>Lines: ${response.content.stats.lineCount}</p>
                            </div>
                        </div>
                    </div>`;
                
                try {
                    const result = await chrome.runtime.sendMessage({
                        action: "analyzeContent",
                        content: response.content.fullText
                    });
                    
                    if (!result.error) {
                        const className = result.prediction.toLowerCase();
                        const analysisDiv = document.createElement('div');
                        analysisDiv.className = `email ${className}`;
                        analysisDiv.innerHTML = `
                            <p><strong>Analysis Result:</strong> ${result.prediction}</p>
                            <p><strong>Confidence:</strong> ${result.confidence}%</p>
                        `;
                        emailsDiv.prepend(analysisDiv);

                        // Display summary if available
                        if (result.summary) {
                            summaryDiv.innerHTML = `
                                <div class="summary-title">Content Summary</div>
                                <div class="summary-content">${result.summary}</div>
                            `;
                        }
                    }
                } catch (analyzeError) {
                    console.error('Analysis error:', analyzeError);
                }
            } else {
                emailsDiv.innerHTML = '<div class="email">No content found on this page</div>';
            }
        });
    } catch (error) {
        console.error('Extension error:', error);
        loadingDiv.style.display = 'none';
        emailsDiv.innerHTML = `<div class="email error">Error: ${error.message}</div>`;
    }
});

// Theme Toggle
document.getElementById('themeToggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');
    document.getElementById('themeToggle').textContent = isDark ? '☀️' : '🌙';
    chrome.storage.local.set({ theme: isDark ? 'dark' : 'light' });
});

// Copy Results
document.getElementById('copyBtn').addEventListener('click', () => {
    const content = document.querySelector('.raw-content').innerText;
    navigator.clipboard.writeText(content)
        .then(() => alert('Content copied to clipboard!'))
        .catch(err => console.error('Failed to copy:', err));
});

// Export Report
document.getElementById('exportBtn').addEventListener('click', async () => {
    const content = document.querySelector('.email').innerText;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'spam-analysis-report.txt';
    a.click();
});

// Save to History
function saveToHistory(result) {
    chrome.storage.local.get({ analysisHistory: [] }, (data) => {
        const history = data.analysisHistory;
        history.unshift({
            timestamp: new Date().toISOString(),
            result: result,
            content: result.content.preview
        });
        // Keep only last 10 entries
        if (history.length > 10) history.pop();
        chrome.storage.local.set({ analysisHistory: history });
    });
}

// View History
document.getElementById('historyBtn').addEventListener('click', () => {
    const historyPanel = document.getElementById('history');
    if (historyPanel.style.display === 'none') {
        chrome.storage.local.get({ analysisHistory: [] }, (data) => {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = data.analysisHistory.map(item => `
                <div class="history-item">
                    <div><strong>${new Date(item.timestamp).toLocaleString()}</strong></div>
                    <div>Result: ${item.result.prediction} (${item.result.confidence}%)</div>
                </div>
            `).join('');
        });
        historyPanel.style.display = 'block';
    } else {
        historyPanel.style.display = 'none';
    }
});
