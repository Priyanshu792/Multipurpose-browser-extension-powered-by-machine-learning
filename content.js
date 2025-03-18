function extractPageContent() {
    console.group('Content Extraction');

    // Gmail-specific selectors to exclude
    const gmailUISelectors = [
        '.TK', // Gmail toolbar
        '.nH.oy8Mbf', // Gmail navigation
        '.gb_Td', // Gmail header
        '[role="banner"]',
        '[role="navigation"]',
        '.a3s.aiL', // Gmail UI elements
        '.gE.iv.gt', // Gmail UI elements
        '.gmail_default', // Default Gmail elements
        '.gmail_quote', // Quoted text in Gmail
        '.ii.gt', // Gmail message thread
        'div[aria-label="Inbox"]',
        'div[aria-label="Main menu"]',
        '.J-M.jQjAxd', // Gmail dropdown menus
    ];

    // Clone body to avoid modifying original
    const bodyClone = document.body.cloneNode(true);

    // Remove Gmail UI elements
    gmailUISelectors.forEach(selector => {
        const elements = bodyClone.querySelectorAll(selector);
        elements.forEach(el => el.remove());
    });

    // Get the actual email content
    let text = '';
    
    // Try Gmail message content first
    const emailContent = document.querySelector('.a3s.aiL');
    if (emailContent) {
        text = emailContent.innerText;
    } else {
        // Try main content area
        const mainContent = document.querySelector('[role="main"]');
        if (mainContent) {
            text = mainContent.innerText;
        } else {
            // Fallback to all visible text content
            const visibleText = Array.from(bodyClone.querySelectorAll('*'))
                .filter(el => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && 
                           style.visibility !== 'hidden' && 
                           el.innerText.trim().length > 0;
                })
                .map(el => el.innerText.trim())
                .join('\n');
            text = visibleText;
        }
    }

    // Clean the text
    const cleanContent = text
        .replace(/[\n\r]+/g, '\n')         // Normalize line breaks
        .replace(/\s+/g, ' ')              // Normalize spaces
        .replace(/Skip to content.*?Inbox/i, '') // Remove Gmail header
        .replace(/\[\[.*?\]\]/g, '')       // Remove [[...]] patterns
        .replace(/\{.*?\}/g, '')           // Remove {...} patterns
        .replace(/\(.*?\)/g, '')           // Remove (...) patterns
        .replace(/Labels.*?More/g, '')     // Remove Gmail labels section
        .trim();

    // Log complete content without truncation
    console.log('=== START OF EXTRACTED CONTENT ===');
    console.log(cleanContent);
    console.log('=== END OF EXTRACTED CONTENT ===');
    console.log('Content Statistics:', {
        'Total Characters': cleanContent.length,
        'Word Count': cleanContent.split(/\s+/).length,
        'Line Count': cleanContent.split('\n').length
    });
    
    console.groupEnd();
    
    return {
        content: {
            fullText: cleanContent,
            preview: cleanContent,  // Send full content instead of preview
            success: true,
            stats: {
                length: cleanContent.length,
                wordCount: cleanContent.split(/\s+/).length,
                lineCount: cleanContent.split('\n').length
            }
        }
    };
}

// Listen for content request
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getPageContent") {
        sendResponse(extractPageContent());
    }
    return true;
});
