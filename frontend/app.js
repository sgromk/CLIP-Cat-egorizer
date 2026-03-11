/**
 * app.js — Art Grading Agent frontend logic
 *
 * Handles:
 *  - File input + drag-and-drop
 *  - Image preview
 *  - POST /grade with FormData
 *  - Rendering grade, explanation, and detected labels
 */

const dropArea     = document.getElementById('drop-area');
const fileInput    = document.getElementById('file-input');
const previewImg   = document.getElementById('preview-img');
const previewCont  = document.getElementById('preview-container');
const gradeBtn     = document.getElementById('grade-btn');
const uploadSec    = document.getElementById('upload-section');
const resultSec    = document.getElementById('result-section');
const loadingEl    = document.getElementById('loading');
const errorSec     = document.getElementById('error-section');
const errorMsg     = document.getElementById('error-message');
const gradeValue   = document.getElementById('grade-value');
const explanEl     = document.getElementById('explanation');
const labelsList   = document.getElementById('labels-list');
const resetBtn     = document.getElementById('reset-btn');

let selectedFile = null;

// -----------------------------------------------------------------------
// File selection ─ input
// -----------------------------------------------------------------------
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// -----------------------------------------------------------------------
// Drag-and-drop
// -----------------------------------------------------------------------
['dragenter', 'dragover'].forEach(evt =>
  dropArea.addEventListener(evt, e => {
    e.preventDefault();
    dropArea.classList.add('dragging');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropArea.addEventListener(evt, e => {
    e.preventDefault();
    dropArea.classList.remove('dragging');
  })
);

dropArea.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// Clicking drop area opens file picker
dropArea.addEventListener('click', () => fileInput.click());

// -----------------------------------------------------------------------
// Handle selected file
// -----------------------------------------------------------------------
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select an image file.');
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewCont.classList.remove('hidden');
  gradeBtn.disabled = false;
  hideError();
}

// -----------------------------------------------------------------------
// Grade button
// -----------------------------------------------------------------------
gradeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/grade', { method: 'POST', body: formData });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail ?? 'Unknown error');
    }

    const data = await resp.json();
    showResult(data);
  } catch (err) {
    showError(`Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// -----------------------------------------------------------------------
// Reset
// -----------------------------------------------------------------------
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewCont.classList.add('hidden');
  gradeBtn.disabled = true;
  resultSec.classList.add('hidden');
  uploadSec.classList.remove('hidden');
  hideError();
});

// -----------------------------------------------------------------------
// UI helpers
// -----------------------------------------------------------------------
function showResult(data) {
  uploadSec.classList.add('hidden');

  gradeValue.textContent = data.grade;
  explanEl.textContent = data.explanation;

  labelsList.innerHTML = '';
  (data.detected_labels ?? []).forEach(([label, score]) => {
    const li = document.createElement('li');
    li.textContent = `${label} (${(score * 100).toFixed(0)}%)`;
    labelsList.appendChild(li);
  });

  resultSec.classList.remove('hidden');
}

function setLoading(active) {
  loadingEl.classList.toggle('hidden', !active);
  gradeBtn.disabled = active;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorSec.classList.remove('hidden');
}

function hideError() {
  errorSec.classList.add('hidden');
}
