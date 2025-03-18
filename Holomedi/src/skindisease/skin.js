import Url from '../url/url.js';

const imageUpload = document.getElementById('file-input');
const uploadIcon = document.getElementById('upload-icon');
const uploadText = document.getElementById('upload-text');
const orContainer = document.getElementById('or-container');
const captureContainer = document.getElementById('capture-container');
const imagePreview = document.getElementById('image-preview');
const captureBtn = document.getElementById('capture-btn');
const cameraContainer = document.getElementById('camera-container');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const submitButton = document.querySelector('.upload-button');
const diseaseResult = document.getElementById('disease-result');
const infoBox = document.getElementById('info-box');
const chatBox = document.getElementById('chat-box');
const cancelChatboxIcon = document.getElementById('cancel-chatbox-icon');

imageUpload.addEventListener('change', (event) => {
  const file = event.target.files[0]; 
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageDataUrl = e.target.result; 
      showImagePreview(imageDataUrl);
    };
    reader.readAsDataURL(file); 
  }
});

captureContainer.addEventListener('click', () => {
  hideNestedBoxElements();
  cameraContainer.style.display = 'block';

  // Activate the camera
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((err) => {
      console.error('Error accessing the camera:', err);
      alert('Unable to access the camera.');
      resetUploadView(); 
    });
});

// Event listener for capture button
captureBtn.addEventListener('click', () => {
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  // Draw the current video frame onto the canvas
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Stop the camera stream
  const stream = video.srcObject;
  const tracks = stream.getTracks();
  tracks.forEach((track) => track.stop());
  // Get the image directly as a Blob from the canvas (instead of using data URL)
  canvas.toBlob((blob) => {
    const file = new File([blob], 'captured-image.png', { type: 'image/png' });
    // Hide the camera and show the preview
    cameraContainer.style.display = 'none';
    showImagePreview(URL.createObjectURL(file)); // Show the captured image preview
    // Store the file globally to use later in the FormData
    window.capturedImageFile = file;
  }, 'image/png');
});

// Function to show the image preview
function showImagePreview(imageDataUrl) {
  // Hide all unnecessary elements
  uploadIcon.style.display = 'none';
  uploadText.style.display = 'none';
  orContainer.style.display = 'none';
  captureContainer.style.display = 'none';

  // Display the image preview with a cancel icon
  imagePreview.innerHTML = `
    <div class="preview-container">
      <img src="${imageDataUrl}" alt="Captured Image" class="uploaded-image">
      <span class="cancel-icon" id="cancel-icon">&times;</span>
    </div>
  `;

  // Add event listener for the cancel icon
  const cancelIcon = document.getElementById('cancel-icon');
  cancelIcon.addEventListener('click', () => {
    resetUploadView();
  });
}

// Function to hide elements inside the nested box
function hideNestedBoxElements() {
  uploadIcon.style.display = 'none';
  uploadText.style.display = 'none';
  orContainer.style.display = 'none';
  captureContainer.style.display = 'none';
}

// Function to reset the upload view
function resetUploadView() {
  // Clear the image preview area
  imagePreview.innerHTML = '';

  // Restore all the hidden elements
  uploadIcon.style.display = 'block';
  uploadText.style.display = 'block';
  orContainer.style.display = 'block';
  captureContainer.style.display = 'block';

  // Hide the camera container
  cameraContainer.style.display = 'none';

  // Stop the camera stream if active
  if (video.srcObject) {
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
  }

  // Clear the file input value
  imageUpload.value = '';
  diseaseResult.textContent = '';
}

submitButton.addEventListener('click', async () => {
  // Check if an image was uploaded
  const imageFile = imageUpload.files[0];
  // If no image is uploaded, use the captured image (stored as Blob)
  let fileToSend = imageFile || window.capturedImageFile;
  if (!fileToSend) {
    alert('Please upload or capture an image first.');
    return;
  }
  diseaseResult.textContent = 'Detecting disease...';
  try {
    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', fileToSend);
    const skinUrl = Url.Base_url + Url.skin;
    // Send the file to the backend
    const response = await fetch(skinUrl, {
      method: 'POST',
      body: formData, 
    });
    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }
    const result = await response.json();
    const detectedDisease = result.predicted_class; // Adjust based on the actual API response structure
    // Display the detected disease
    diseaseResult.innerHTML = `
      <p>Detected Disease: ${detectedDisease}</p>
      <a href="#" id="details-link" class="details-link">Details</a>
    `;
    // Add event listener for the "Details" link
    const detailsLink = document.getElementById('details-link');
    detailsLink.addEventListener('click', (event) => {
      event.preventDefault(); 
      showChatBox(detectedDisease); 
    });
  } catch (error) {
    console.error('Error detecting disease:', error);
    diseaseResult.textContent = 'Upload a valid Image.';
  }
});

// Show the chat box with a tilt animation
// Show the chat box with a tilt animation and send a dynamic message based on detected disease
function showChatBox(diseaseName) {
  const messagesContainer = document.getElementById('messages-container');
  messagesContainer.innerHTML = ''; 

  infoBox.style.display = 'none';

  // Show the chat box with the tilt animation
  chatBox.style.display = 'block';
  setTimeout(() => {
    chatBox.classList.add('visible'); 
  }, 0);

  // Automatically send a dynamic message to the chatbot backend
  const messageInput = document.getElementById('message-input');
  messageInput.value = `Explain about ${diseaseName}?`; 

  // Trigger the send message event
  const sendIcon = document.getElementById('send-icon');
  sendIcon.click(); 
}

function showInfoBox() {
  const messagesContainer = document.getElementById('messages-container');
  messagesContainer.innerHTML = ''; 

  chatBox.classList.remove('visible');
  setTimeout(() => {
    chatBox.style.display = 'none';
    infoBox.style.display = 'block'; 
  }, 300); 
}

// Event listener for the cancel icon
cancelChatboxIcon.addEventListener('click', () => {
  showInfoBox();
});



document.addEventListener('DOMContentLoaded', function () {
  const sendIcon = document.getElementById('send-icon');
  const messageInput = document.getElementById('message-input');
  const messagesContainer = document.getElementById('messages-container');
  const micIcon = document.querySelector('.mic-icon');
  const micStatus = document.createElement('div');

  micStatus.classList.add('mic-status');
  micStatus.style.display = 'none';
  micStatus.innerText = 'Listening...';
  document.body.appendChild(micStatus);

  let recognition;
  let isRecording = false;
  let inactivityTimeout;

  // Function to handle sending message
  async function sendMessage() {
      const messageText = messageInput.value.trim();

      if (messageText) {
          // Clear the input field immediately
          messageInput.value = '';

          // Create the user's message bubble
          const userMessage = document.createElement('div');
          userMessage.classList.add('message', 'user-message');
          userMessage.innerText = messageText;
          messagesContainer.appendChild(userMessage);

          const typingMessage = document.createElement('div');
          typingMessage.classList.add('message', 'chatbot-message', 'typing-message');
          typingMessage.innerText = 'Medical assistant is typing...';
          messagesContainer.appendChild(typingMessage);
  
          // Scroll to the bottom of the messages container
          messagesContainer.scrollTop = messagesContainer.scrollHeight;

          try {
              // Make an API call to the Flask backend
              const chatUrl = Url.Base_url + Url.chat;
              const response = await fetch(chatUrl, {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/x-www-form-urlencoded'
                  },
                  body: new URLSearchParams({
                      query: messageText
                  })
              });

              const data = await response.json();
              messagesContainer.removeChild(typingMessage);
              // Create the chatbot's message bubble with the API response
              const chatbotMessage = document.createElement('div');
              chatbotMessage.classList.add('message', 'chatbot-message');
              chatbotMessage.innerText = data.response; // Response from Flask
              messagesContainer.appendChild(chatbotMessage);
          } catch (error) {
              messagesContainer.removeChild(typingMessage);
              const errorMessage = document.createElement('div');
              errorMessage.classList.add('message', 'error-message');
              errorMessage.innerText = 'Error: Unable to connect to the server.';
              messagesContainer.appendChild(errorMessage);
          }

          // Scroll to the bottom of the messages container
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
  }

  // Event listener for Send button click
  sendIcon.addEventListener('click', function () {
      if (isRecording) {
          stopSpeechRecognition();
      }
      sendMessage();
  });

  // Event listener for Enter key press
  messageInput.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') {
          event.preventDefault();
          sendMessage();
      }
  });

  // Function to handle Speech Recognition
  function startSpeechRecognition() {
      if (!('webkitSpeechRecognition' in window)) {
          alert('Your browser does not support Speech Recognition.');
          return;
      }

      recognition = new webkitSpeechRecognition();
      recognition.lang = 'en-US';
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.start();

      micStatus.style.display = 'block';

      function resetInactivityTimer() {
          clearTimeout(inactivityTimeout);
          inactivityTimeout = setTimeout(function () {
              console.log('No speech detected for 3 seconds, stopping recognition.');
              stopSpeechRecognition();
              sendMessage();
          }, 3000);
      }

      recognition.onresult = function (event) {
          const speechToText = event.results[event.results.length - 1][0].transcript;
          messageInput.value = speechToText;
          resetInactivityTimer();
      };

      recognition.onerror = function (event) {
          console.error('Speech recognition error: ', event.error);
          micStatus.style.display = 'none';
      };

      recognition.onend = function () {
          micStatus.style.display = 'none';
          if (isRecording) {
              stopSpeechRecognition();
              sendMessage();
          }
      };
  }

  function stopSpeechRecognition() {
      if (recognition) {
          recognition.stop();
          micStatus.style.display = 'none';
          clearTimeout(inactivityTimeout);
      }
  }

  micIcon.addEventListener('click', function () {
      if (!isRecording) {
          startSpeechRecognition();
      }
      isRecording = true;
  });

  window.toggleMenu = function() {
    var menu = document.getElementById('navLinks');
    if (menu.style.display === 'flex') {
        menu.style.display = 'none';
    } else {
        menu.style.display = 'flex';
    }
};

// Close the menu after a link is clicked on mobile
document.querySelectorAll('.nav-links a').forEach(function(link) {
    link.addEventListener('click', function() {
        var menu = document.getElementById('navLinks');
        menu.style.display = 'none'; // Close the menu
    });
});

});
