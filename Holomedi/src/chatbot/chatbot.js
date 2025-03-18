
import Url  from '../url/url.js';

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
                chatbotMessage.innerText = data.response; 
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
  