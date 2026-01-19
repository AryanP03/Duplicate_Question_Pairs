/**
 * ============================================
 * QUORA QUESTION SIMILARITY CHECKER
 * JavaScript Logic for NLP-style Similarity Detection
 * ============================================
 */

// DOM Elements
const question1Input = document.getElementById('question1');
const question2Input = document.getElementById('question2');
const error1Element = document.getElementById('error1');
const error2Element = document.getElementById('error2');
const compareBtn = document.getElementById('compareBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const clearBtn = document.getElementById('clearBtn');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultDescription = document.getElementById('resultDescription');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');

// SVG Icons
const checkIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
const xIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;

/**
 * Normalize text for NLP processing
 * - Convert to lowercase
 * - Remove punctuation
 * - Split into words
 * - Filter out short words (less than 3 characters)
 * @param {string} text - Input text to normalize
 * @returns {string[]} Array of normalized words
 */
function normalizeText(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 2);
}

/**
 * Calculate similarity between two questions
 * Uses a combination of Jaccard similarity and Cosine similarity
 * @param {string} q1 - First question
 * @param {string} q2 - Second question
 * @returns {{ isSimilar: boolean, confidence: number }}
 */
function calculateSimilarity(q1, q2) {
  const words1 = normalizeText(q1);
  const words2 = normalizeText(q2);

  // Handle empty input
  if (words1.length === 0 || words2.length === 0) {
    return { isSimilar: false, confidence: 0 };
  }

  // Create sets for Jaccard similarity
  const set1 = new Set(words1);
  const set2 = new Set(words2);

  // Calculate intersection and union
  const intersection = [...set1].filter(word => set2.has(word));
  const union = new Set([...set1, ...set2]);

  // Jaccard similarity: intersection / union
  const jaccardSimilarity = intersection.length / union.size;

  // Calculate word frequency vectors for Cosine similarity
  const allWords = [...union];
  const freq1 = allWords.map(w => words1.filter(x => x === w).length);
  const freq2 = allWords.map(w => words2.filter(x => x === w).length);

  // Cosine similarity: dot product / (magnitude1 * magnitude2)
  const dotProduct = freq1.reduce((sum, val, i) => sum + val * freq2[i], 0);
  const magnitude1 = Math.sqrt(freq1.reduce((sum, val) => sum + val * val, 0));
  const magnitude2 = Math.sqrt(freq2.reduce((sum, val) => sum + val * val, 0));
  const cosineSimilarity = (magnitude1 && magnitude2) 
    ? dotProduct / (magnitude1 * magnitude2) 
    : 0;

  // Combined score with weighted average (40% Jaccard, 60% Cosine)
  const combinedScore = jaccardSimilarity * 0.4 + cosineSimilarity * 0.6;
  const confidence = Math.min(Math.round(combinedScore * 100), 100);

  return {
    isSimilar: confidence >= 40,
    confidence: confidence
  };
}

/**
 * Validate inputs and show error messages
 * @returns {boolean} True if inputs are valid
 */
function validateInputs() {
  let isValid = true;

  // Clear previous errors
  error1Element.textContent = '';
  error2Element.textContent = '';

  // Validate Question 1
  if (!question1Input.value.trim()) {
    error1Element.textContent = 'Please enter Question 1';
    isValid = false;
  }

  // Validate Question 2
  if (!question2Input.value.trim()) {
    error2Element.textContent = 'Please enter Question 2';
    isValid = false;
  }

  return isValid;
}

/**
 * Display the result with animations
 * @param {{ isSimilar: boolean, confidence: number }} result
 */
function displayResult(result) {
  // Show result card
  resultCard.classList.remove('hidden');

  // Set icon and styling
  if (result.isSimilar) {
    resultIcon.innerHTML = checkIcon;
    resultIcon.className = 'result-icon success';
    resultTitle.textContent = 'The questions are similar';
    resultDescription.textContent = 'These questions appear to have the same meaning or intent.';
  } else {
    resultIcon.innerHTML = xIcon;
    resultIcon.className = 'result-icon failure';
    resultTitle.textContent = 'The questions are not similar';
    resultDescription.textContent = 'These questions seem to be asking about different things.';
  }

  // Animate confidence meter
  confidenceValue.textContent = '0%';
  confidenceFill.style.width = '0%';

  // Use requestAnimationFrame for smooth animation
  setTimeout(() => {
    confidenceFill.style.width = result.confidence + '%';
    animateConfidenceValue(0, result.confidence, 1000);
  }, 100);
}

/**
 * Animate the confidence percentage number
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} duration - Animation duration in ms
 */
function animateConfidenceValue(start, end, duration) {
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // Easing function (ease-out cubic)
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const currentValue = Math.round(start + (end - start) * easeOut);

    confidenceValue.textContent = currentValue + '%';

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

/**
 * Set loading state on button
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
  if (isLoading) {
    compareBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('hidden');
    // Hide the sparkle icon
    compareBtn.querySelector('.icon').classList.add('hidden');
  } else {
    compareBtn.disabled = false;
    btnText.textContent = 'Compare Questions';
    spinner.classList.add('hidden');
    compareBtn.querySelector('.icon').classList.remove('hidden');
  }
}

/**
 * Update visibility of clear button
 */
function updateClearButton() {
  const hasContent = question1Input.value || question2Input.value || !resultCard.classList.contains('hidden');
  if (hasContent) {
    clearBtn.classList.remove('hidden');
  } else {
    clearBtn.classList.add('hidden');
  }
}

/**
 * Clear all inputs and results
 */
function clearAll() {
  question1Input.value = '';
  question2Input.value = '';
  error1Element.textContent = '';
  error2Element.textContent = '';
  resultCard.classList.add('hidden');
  clearBtn.classList.add('hidden');
}

/**
 * Handle compare button click
 */
async function handleCompare() {
  // Hide any previous result
  resultCard.classList.add('hidden');

  // Validate input fields
  if (!validateInputs()) {
    updateClearButton();
    return;
  }

  // Show loading state
  setLoading(true);
  updateClearButton();

  try {
    // 🔥 SEND USER INPUT TO BACKEND
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question1: question1Input.value,
        question2: question2Input.value
      })
    });

    // Convert backend response to JS object
    const result = await response.json();

    // 🔥 DISPLAY MODEL OUTPUT
    displayResult({
      isSimilar: result.is_similar,
      confidence: result.confidence
    });

  } catch (error) {
    console.error("Backend error:", error);
    alert("Model server is not running or unreachable");
  }

  // Reset loading state
  setLoading(false);
  updateClearButton();
}


// Event Listeners
compareBtn.addEventListener('click', handleCompare);
clearBtn.addEventListener('click', clearAll);

// Clear error on input
question1Input.addEventListener('input', () => {
  if (error1Element.textContent) {
    error1Element.textContent = '';
  }
  updateClearButton();
});

question2Input.addEventListener('input', () => {
  if (error2Element.textContent) {
    error2Element.textContent = '';
  }
  updateClearButton();
});

// Allow Enter key to trigger comparison (with Ctrl/Cmd)
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    handleCompare();
  }
});

// Initialize
updateClearButton();
