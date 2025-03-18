
import Url from "../url/url.js";

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const uploadText = document.getElementById("upload-text");
  const uploadIcon = document.getElementById("upload-icon");
  const sendIcon = document.getElementById("send-icon");
  const messageInput = document.getElementById("message-input");
  const diseaseResult = document.getElementById("disease-result");
  const imagePreview = document.getElementById("image-preview");
  // Create a loading indicator using a <div> element
  const loadingIndicator = document.createElement("div");
  loadingIndicator.textContent = "Analyzing image...";
  loadingIndicator.classList.add("loading-indicator");
  // Function to show image preview
  function showImagePreview(imageDataUrl) {
    uploadIcon.style.display = "none";
    uploadText.style.display = "none"; 
    imagePreview.innerHTML = `
      <div class="preview-container">
        <img src="${imageDataUrl}" alt="Uploaded Im
        age" class="preview-image">
        <span class="cancel-icon" id="cancel-icon">&times;</span>
      </div>
    `;
    const cancelIcon = document.getElementById("cancel-icon");
    cancelIcon.addEventListener("click", resetToHomeScreen);
  }
  // Function to reset to the home screen
  function resetToHomeScreen() {
    imagePreview.innerHTML = "";
    diseaseResult.innerHTML = "";
    fileInput.value = "";
    
    // Reset upload text and icon
    uploadText.style.display = "block";
    uploadIcon.style.display = "block";
  
    // Reset icon position explicitly (in case of unexpected shifts)
    uploadIcon.style.top = "50%"; 
    uploadIcon.style.left = "50%"; 
    loadingIndicator.style.display = "none";
    messageInput.value = "";
  }
  
  // Handle image upload event
  fileInput.addEventListener("change", (event) => {
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
  // Handle the send button click event
  async function handleSubmit() {
    const file = fileInput.files[0];
    const question = messageInput.value.trim();
    if (!file) {
      alert('Please upload an image.');
      return;
    }
    const allowedTypes = ["image/jpeg", "image/png", "image/jpg", "image/WEBP"];
    if (!allowedTypes.includes(file.type)) {
      alert('Error: Unsupported file type');
      return;
    }
   
    loadingIndicator.style.display = "block";
    diseaseResult.innerHTML = "";
    imagePreview.appendChild(loadingIndicator);
    messageInput.value = "";
    
    const formData = new FormData();
    formData.append("image", file);
    formData.append("question", question);

    const xrayUrl = Url.Base_url1 + Url.xray;

    try {
      const response = await fetch(xrayUrl, {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        const { table } = data;
        diseaseResult.innerHTML = `
          <p><strong>Analysis result:</strong> ${table || "No result generated."}</p>`;
      } else {
        const errorData = await response.json();
  
        diseaseResult.innerHTML = `<p class="error-message">Error: ${errorData.error || "Bad Request"}</p>`;
      }
    } catch (error) {

      diseaseResult.innerHTML = `<p class="error-message">An error occurred while connecting to the backend.</p>`;
    } finally {
      loadingIndicator.style.display = "none";
      
    }
  }
    sendIcon.addEventListener("click", handleSubmit);

    // Handle the "Enter" key press event
    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault(); 
        handleSubmit();
      }
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