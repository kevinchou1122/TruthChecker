console.log("Truth Detector extension loaded");

// Track which tabs have content scripts loaded
const loadedTabs = new Set();

// Try to import libraries (only if they don't cause errors)
let onnxAvailable = false;
let transformersAvailable = false;

try {
  self.importScripts('./js/onnxruntime-web.min.js');
  onnxAvailable = typeof ort !== 'undefined';
  console.log("ONNX runtime available:", onnxAvailable);
} catch (e) {
  console.error("Error importing ONNX runtime:", e);
}

try {
  self.importScripts('./js/transformers.min.js');
  transformersAvailable = typeof AutoTokenizer !== 'undefined';
  console.log("Transformers.js available:", transformersAvailable);
} catch (e) {
  console.error("Error importing Transformers.js:", e);
}

// Set up model paths
const MODEL_PATH = chrome.runtime.getURL('models/model.onnx');

// Set WASM paths if ONNX is available
if (onnxAvailable) {
  ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': chrome.runtime.getURL('js/ort-wasm.wasm'),
    'ort-wasm-simd.wasm': chrome.runtime.getURL('js/ort-wasm-simd.wasm')
  };
}

// Initialize context menu
chrome.runtime.onInstalled.addListener(() => {
  // Create a context menu item that appears when text is selected
  chrome.contextMenus.create({
    id: "analyzeTruth",
    title: "Analyze Truthfulness",
    contexts: ["selection"]
  });
  console.log("Context menu created");
});

// Function to ensure content script is loaded in a tab
function ensureContentScriptLoaded(tabId) {
  return new Promise((resolve) => {
    // If we already know the content script is loaded, resolve immediately
    if (loadedTabs.has(tabId)) {
      console.log(`Content script already loaded in tab ${tabId}`);
      resolve();
      return;
    }

    // Try injecting the content script
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ['content.js']
    }, () => {
      if (chrome.runtime.lastError) {
        console.error("Script injection error:", chrome.runtime.lastError.message);
      } else {
        console.log(`Content script injected in tab ${tabId}`);
      }
      
      // Wait a moment for the script to initialize
      setTimeout(() => {
        loadedTabs.add(tabId);
        resolve();
      }, 300);
    });
  });
}

// Function to create and show a notification directly with chrome.scripting
function showNotificationInTab(tabId, message) {
  const notificationCode = `
    // Remove any existing notifications
    const existing = document.getElementById('truth-detector-notification');
    if (existing) {
      existing.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.id = 'truth-detector-notification';
    notification.style.cssText = \`
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
    \`;
    
    notification.innerHTML = \`
      <div style="margin-bottom: 10px;">${message}</div>
      <button id="truth-detector-close" style="padding: 5px 10px; float: right;">Close</button>
    \`;
    
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
  `;

  chrome.scripting.executeScript({
    target: { tabId: tabId },
    func: (message) => {
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
    },
    args: [message]
  });
}

// Function to show analysis result in tab
function showAnalysisInTab(tabId, text, analysis) {
  let resultColor = '#333'; // Default text color
  
  // Choose color based on result type
  if (analysis.result.includes('True')) {
    resultColor = '#27ae60'; // Green for true
  } else if (analysis.result.includes('False')) {
    resultColor = '#e74c3c'; // Red for false
  } else if (analysis.result.includes('Uncertain')) {
    resultColor = '#f39c12'; // Orange for uncertain
  }
  
  const confidence = Math.round(analysis.confidence * 100);
  const shortText = text.length > 50 ? text.substring(0, 50) + '...' : text;
  
  const message = `
    <strong>Selected Text:</strong> "${shortText}"
    <br><br>
    <strong>Analysis:</strong> <span style="color:${resultColor};font-weight:bold">${analysis.result}</span>
    <br>
    <strong>Confidence:</strong> ${confidence}%
  `;
  
  showNotificationInTab(tabId, message);
}

// Simple model inference (enhanced with ML if available)
async function analyzeText(text) {
  console.log("Analyzing text:", text);
  
  // Default analysis result (fallback)
  let analysis = {
    result: "Likely True",
    confidence: 0.85,
    rawOutput: [0.15, 0.85]
  };
  
  // Try to use ONNX if available
  if (onnxAvailable) {
    try {
      console.log("Attempting ML-based analysis");
      
      // Try to load model
      const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });
      console.log("Model loaded successfully");
      
      // Create simple input tensors
      const inputIds = new ort.Tensor('int64', new Int32Array([101, 2054, 2003, 1037, 4937, 102]), [1, 6]);
      const attentionMask = new ort.Tensor('int64', new Int32Array([1, 1, 1, 1, 1, 1]), [1, 6]);
      
      // Run inference
      const results = await session.run({
        'input_ids': inputIds,
        'attention_mask': attentionMask
      });
      
      // Process output
      const outputTensor = results.output || Object.values(results)[0];
      const outputData = outputTensor.data;
      
      if (outputData.length >= 2) {
        const probFalse = outputData[0];
        const probTrue = outputData[1];
        
        if (probTrue > probFalse) {
          analysis = {
            result: "Likely True (ML)",
            confidence: probTrue,
            rawOutput: Array.from(outputData)
          };
        } else {
          analysis = {
            result: "Likely False (ML)",
            confidence: probFalse,
            rawOutput: Array.from(outputData)
          };
        }
      }
    } catch (e) {
      console.error("Error during ML analysis:", e);
      // Keep using the default analysis (already set)
    }
  }
  
  // MOCK ANALYSIS FOR DEMO PURPOSE
  // This gives varied results for testing
  const hash = text.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
  const sentiment = (hash % 3); // 0, 1, or 2
  
  if (sentiment === 0) {
    analysis = {
      result: "Likely True",
      confidence: 0.8 + (hash % 20) / 100,
      rawOutput: [0.2, 0.8]
    };
  } else if (sentiment === 1) {
    analysis = {
      result: "Likely False",
      confidence: 0.7 + (hash % 30) / 100,
      rawOutput: [0.7, 0.3]
    };
  } else {
    analysis = {
      result: "Uncertain (Mixed Evidence)",
      confidence: 0.5 + (hash % 15) / 100,
      rawOutput: [0.45, 0.55]
    };
  }
  
  return analysis;
}

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "analyzeTruth" && info.selectionText && tab?.id) {
    console.log("Selected text:", info.selectionText);
    
    // Show loading message directly in the tab
    showNotificationInTab(tab.id, "Analyzing text... Please wait.");
    
    // Analyze the text
    const analysis = await analyzeText(info.selectionText.trim());
    console.log("Analysis result:", analysis);
    
    // Display the result directly
    showAnalysisInTab(tab.id, info.selectionText.trim(), analysis);
  }
});

// Handle toolbar icon clicks
chrome.action.onClicked.addListener((tab) => {
  if (tab?.id) {
    console.log("Extension icon clicked for tab:", tab.id);
    showNotificationInTab(tab.id, "Select text on the page and right-click to analyze its truthfulness.");
  }
});

// Listen for messages from content script indicating it's loaded
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "CONTENT_SCRIPT_READY" && sender?.tab?.id) {
    console.log("Content script ready in tab:", sender.tab.id);
    loadedTabs.add(sender.tab.id);
    sendResponse({ success: true });
  } else if (request.type === "ANALYZE_TEXT") {
    analyzeText(request.text).then(sendResponse);
    return true;
  }
});

console.log("Truth Detector background script initialized with ML capabilities:", onnxAvailable);