// Simple content script for the Truth Detector extension
console.log("Truth Detector content script loaded");

// Show a notification popup
function showNotification(message) {
  // Remove any existing notifications
  const existing = document.getElementById('truth-detector-notification');
  if (existing) {
    existing.remove();
  }
  
  // Create notification element
  const notification = document.createElement('div');
  notification.id = 'truth-detector-notification';
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px;
    background-color: white;
    border: 1px solid #ccc;
    border-radius: 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    z-index: 10000;
    font-family: Arial, sans-serif;
    max-width: 300px;
  `;
  
  notification.innerHTML = `
    <div style="margin-bottom: 10px;">${message}</div>
    <button id="truth-detector-close" style="padding: 5px 10px; float: right;">Close</button>
  `;
  
  document.body.appendChild(notification);
  
  // Add close button handler
  document.getElementById('truth-detector-close').addEventListener('click', () => {
    notification.remove();
  });
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (document.getElementById('truth-detector-notification')) {
      notification.remove();
    }
  }, 10000);
}

// Display analysis results with color coding
function displayAnalysisResult(text, analysis) {
  let resultMessage = '';
  let resultColor = '#333'; // Default text color
  
  if (analysis.error) {
    resultMessage = `Error: ${analysis.error}`;
    resultColor = '#e74c3c'; // Red for errors
  } else {
    // Choose color based on result type
    if (analysis.result.includes('True')) {
      resultColor = '#27ae60'; // Green for true
    } else if (analysis.result.includes('False')) {
      resultColor = '#e74c3c'; // Red for false
    } else if (analysis.result.includes('Uncertain')) {
      resultColor = '#f39c12'; // Orange for uncertain
    }
    
    const confidence = Math.round(analysis.confidence * 100);
    
    resultMessage = `
      <strong>Selected Text:</strong> "${text.length > 50 ? text.substring(0, 50) + '...' : text}"
      <br><br>
      <strong>Analysis:</strong> <span style="color:${resultColor};font-weight:bold">${analysis.result}</span>
      <br>
      <strong>Confidence:</strong> ${confidence}%
    `;
  }
  
  showNotification(resultMessage);
}

// Let the background script know the content script is ready
chrome.runtime.sendMessage({ type: "CONTENT_SCRIPT_READY" }, (response) => {
  if (chrome.runtime.lastError) {
    console.error("Error sending ready message:", chrome.runtime.lastError.message);
  } else {
    console.log("Sent ready message to background script");
  }
});

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Content script received message:", message);
  
  if (message.type === "TRUTH_ANALYSIS_RESULT") {
    displayAnalysisResult(message.selectedText, message.data);
    sendResponse({ success: true });
  } else if (message.type === "SHOW_INFO") {
    showNotification(message.data.message);
    sendResponse({ success: true });
  }
  
  return true; // Indicates we'll respond asynchronously
});

// Alert when loaded (for debugging)
console.log("Truth Detector content script initialized and ready");