// API Configuration
const API_BASE_URL = 'http://localhost:5000';
let currentImage = null;

// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const notification = document.getElementById('notification');
const notificationText = document.getElementById('notificationText');
const apiStatus = document.getElementById('apiStatus');
const classesGrid = document.getElementById('classesGrid');

// Results Elements
const diseaseName = document.getElementById('diseaseName');
const confidenceScore = document.getElementById('confidenceScore');
const plantType = document.getElementById('plantType');
const healthStatus = document.getElementById('healthStatus');
const analysisTime = document.getElementById('analysisTime');
const adviceText = document.getElementById('adviceText');
const plantIcon = document.getElementById('plantIcon');
const predictionsList = document.getElementById('predictionsList');
const uploadNewBtn = document.getElementById('uploadNewBtn');
const testApiBtn = document.getElementById('testApiBtn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    checkApiConnection();
    loadDetectableClasses();
    setupEventListeners();
});

// Check API Connection
async function checkApiConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            apiStatus.textContent = 'Connected ✓';
            apiStatus.style.color = '#a5d6a7';
            showNotification('API connected successfully', 'success');
        } else {
            apiStatus.textContent = 'Connection Failed ✗';
            apiStatus.style.color = '#ef9a9a';
            showNotification('API connection failed. Make sure backend is running on port 5000.', 'error');
        }
    } catch (error) {
        apiStatus.textContent = 'Not Connected ✗';
        apiStatus.style.color = '#ef9a9a';
        console.error('API Connection Error:', error);
    }
}

// Load detectable classes from API
async function loadDetectableClasses() {
    try {
        const response = await fetch(`${API_BASE_URL}/classes`);
        const data = await response.json();
        
        if (data.classes && data.classes.length > 0) {
            renderClasses(data.classes);
        }
    } catch (error) {
        console.error('Error loading classes:', error);
        classesGrid.innerHTML = `
            <div class="loading-classes">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Failed to load detectable diseases. Please check API connection.</p>
            </div>
        `;
    }
}

// Render classes in the grid
function renderClasses(classes) {
    classesGrid.innerHTML = '';
    
    classes.forEach(cls => {
        const classCard = document.createElement('div');
        classCard.className = `class-card ${cls.is_healthy ? 'healthy' : ''}`;
        
        // Get appropriate icon
        let icon = 'fa-leaf';
        if (cls.plant.includes('Pepper')) icon = 'fa-pepper-hot';
        if (cls.plant.includes('Potato')) icon = 'fa-potato';
        if (cls.plant.includes('Tomato')) icon = 'fa-apple-alt';
        
        classCard.innerHTML = `
            <div class="class-header">
                <div class="class-icon">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="class-name">${cls.plant} ${cls.type}</div>
            </div>
            <div class="class-details">
                <p><strong>Disease:</strong> ${cls.disease}</p>
                <p><strong>Status:</strong> ${cls.is_healthy ? 'Healthy Plant' : 'Diseased'}</p>
            </div>
            <div class="class-status ${cls.is_healthy ? 'status-healthy' : 'status-diseased'}">
                ${cls.is_healthy ? 'HEALTHY' : 'DISEASED'}
            </div>
        `;
        
        classesGrid.appendChild(classCard);
    });
}

// Setup event listeners
function setupEventListeners() {
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Click upload area to trigger file input
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // Analyze button click
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Upload new button
    uploadNewBtn.addEventListener('click', resetUpload);
    
    // Test API button
    testApiBtn.addEventListener('click', checkApiConnection);
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && validateFile(file)) {
        currentImage = file;
        previewImage(file);
        analyzeBtn.disabled = false;
        showNotification('Image uploaded successfully', 'success');
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && validateFile(file)) {
        currentImage = file;
        previewImage(file);
        analyzeBtn.disabled = false;
        showNotification('Image uploaded successfully', 'success');
    }
}

// Validate file
function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const maxSize = 5 * 1024 * 1024; // 5MB
    
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image file (JPG, PNG, BMP, GIF)', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File size must be less than 5MB', 'error');
        return false;
    }
    
    return true;
}

// Preview image
function previewImage(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.innerHTML = `
            <img src="${e.target.result}" alt="Preview">
            <div class="image-info">
                <p>${file.name}</p>
                <p>${(file.size / 1024).toFixed(2)} KB</p>
            </div>
        `;
    };
    
    reader.readAsDataURL(file);
}

// Reset upload
function resetUpload() {
    currentImage = null;
    imageInput.value = '';
    imagePreview.innerHTML = `
        <div class="placeholder">
            <i class="fas fa-seedling"></i>
            <p>No image selected</p>
        </div>
    `;
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

// Show loading overlay
function showLoading(message = 'Processing image...') {
    document.getElementById('loadingMessage').textContent = message;
    document.getElementById('progressFill').style.width = '0%';
    loadingOverlay.style.display = 'flex';
    
    // Animate progress bar
    setTimeout(() => {
        document.getElementById('progressFill').style.width = '70%';
    }, 300);
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('progressFill').style.width = '100%';
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
    }, 500);
}

// Show notification
function showNotification(message, type = 'success') {
    notificationText.textContent = message;
    notification.className = `notification ${type}`;
    notification.style.display = 'flex';
    
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

// Analyze image
async function analyzeImage() {
    if (!currentImage) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    showLoading('Analyzing plant image for diseases...');
    
    const formData = new FormData();
    formData.append('file', currentImage);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayResults(data);
            showNotification('Analysis complete! Disease detected successfully.', 'success');
        } else {
            showNotification(data.error || 'Analysis failed. Please try again.', 'error');
        }
        
    } catch (error) {
        hideLoading();
        console.error('Analysis Error:', error);
        showNotification('Network error. Please check API connection.', 'error');
    }
}

// Display results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Update main result
    diseaseName.textContent = data.display_name;
    confidenceScore.textContent = `${data.confidence}%`;
    plantType.textContent = `Plant: ${data.plant_type}`;
    healthStatus.textContent = `Status: ${data.is_healthy ? 'Healthy' : 'Diseased'}`;
    analysisTime.textContent = `Time: ${new Date().toLocaleTimeString()}`;
    adviceText.textContent = data.advice;
    
    // Update plant icon color based on health
    plantIcon.style.background = data.is_healthy 
        ? 'linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)'
        : 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)';
    
    // Update predictions
    updatePredictions(data.top_predictions);
    
    // Update confidence score color based on value
    const confidence = parseFloat(data.confidence);
    if (confidence >= 80) {
        confidenceScore.style.color = '#2e7d32';
    } else if (confidence >= 60) {
        confidenceScore.style.color = '#ff9800';
    } else {
        confidenceScore.style.color = '#f44336';
    }
}

// Update predictions list
function updatePredictions(predictions) {
    predictionsList.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const predictionItem = document.createElement('div');
        predictionItem.className = 'prediction-item';
        
        predictionItem.innerHTML = `
            <div class="prediction-rank">${index + 1}</div>
            <div class="prediction-details">
                <div class="prediction-name">${pred.display_name}</div>
                <div class="prediction-bar">
                    <div class="bar-fill" style="width: ${pred.confidence}%"></div>
                </div>
            </div>
            <div class="prediction-percent">${pred.confidence}%</div>
        `;
        
        predictionsList.appendChild(predictionItem);
    });
}

// Initialize with sample data for testing
function initSampleData() {
    // This is just for testing UI without API
    const sampleData = {
        success: true,
        display_name: "Tomato - Bacterial Spot",
        confidence: 85.5,
        plant_type: "Tomato",
        is_healthy: false,
        advice: "Bacterial spot detected. Remove infected leaves, avoid overhead watering, and consider copper-based fungicides.",
        top_predictions: [
            {
                display_name: "Tomato - Bacterial Spot",
                confidence: 85.5
            },
            {
                display_name: "Tomato - Early Blight",
                confidence: 12.3
            },
            {
                display_name: "Tomato - Healthy",
                confidence: 2.2
            }
        ]
    };
    
    displayResults(sampleData);
}

// For testing - uncomment to use sample data
// initSampleData();