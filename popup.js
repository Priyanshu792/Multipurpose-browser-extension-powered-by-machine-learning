document.addEventListener('DOMContentLoaded', async () => {
    const loadingDiv = document.getElementById('loading');
    const emailsDiv = document.getElementById('emails');
    
    loadingDiv.style.display = 'block';
    emailsDiv.innerHTML = '';

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
