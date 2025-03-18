function toggleDropdown() {
    const dropdownMenu = document.getElementById("dropdownMenu");
    // Toggle the display style between block and none
    dropdownMenu.style.display = dropdownMenu.style.display === "block" ? "none" : "block";
  }

  // Add an event listener to the menu icon
  document.querySelector(".menu-icon").addEventListener("click", toggleDropdown);

  // Close the dropdown if clicked outside
  window.addEventListener("click", function (event) {
    const dropdownMenu = document.getElementById("dropdownMenu");
    const menuIcon = document.querySelector(".menu-icon");

    // If the click is outside the menu icon and dropdown menu, close the dropdown
    if (!menuIcon.contains(event.target) && !dropdownMenu.contains(event.target)) {
      dropdownMenu.style.display = "none";
    }
  });
