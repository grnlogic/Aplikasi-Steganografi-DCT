document.addEventListener("DOMContentLoaded", () => {
    // Tab navigation
    const tabLinks = document.querySelectorAll(".nav-link")
    tabLinks.forEach((link) => {
      link.addEventListener("click", function (e) {
        e.preventDefault()
        const targetId = this.getAttribute("data-bs-target")
  
        // Update active tab
        tabLinks.forEach((l) => l.classList.remove("active"))
        this.classList.add("active")
  
        // Show target tab content
        document.querySelectorAll(".tab-pane").forEach((pane) => {
          pane.classList.remove("show", "active")
        })
        document.querySelector(targetId).classList.add("show", "active")
      })
    })
  
    // Handle embed form submission
    const embedForm = document.getElementById("embedForm")
    if (embedForm) {
      embedForm.addEventListener("submit", (e) => {
        e.preventDefault()
  
        const fileInput = document.getElementById("coverImage")
        const messageInput = document.getElementById("secretMessage")
  
        if (!fileInput.files[0]) {
          showError("Please select an image file.")
          return
        }
  
        if (!messageInput.value.trim()) {
          showError("Please enter a message to hide.")
          return
        }
  
        // Show loading modal
        const loadingModalElement = document.getElementById("loadingModal")
        const loadingModal = new bootstrap.Modal(loadingModalElement)
        document.getElementById("loadingMessage").textContent = "Embedding message..."
        document.getElementById("loadingSubMessage").textContent = "This may take a few moments"
        loadingModal.show()
  
        // Create form data
        const formData = new FormData()
        formData.append("image", fileInput.files[0])
        formData.append("message", messageInput.value)
  
        // Send request to server
        fetch("/embed", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            loadingModal.hide()
  
            if (data.error) {
              showError(data.error)
              return
            }
  
            // Display stego image
            const stegoContainer = document.getElementById("stegoImageContainer")
            stegoContainer.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Stego Image">`
  
            // Show download button
            document.getElementById("downloadContainer").classList.remove("d-none")
  
            // Display metrics
            document.getElementById("metricsPlaceholder").classList.add("d-none")
            document.getElementById("metricsContainer").classList.remove("d-none")
  
            document.getElementById("psnrValue").textContent = data.metrics.psnr
            document.getElementById("mseValue").textContent = data.metrics.mse
            document.getElementById("ssimValue").textContent = data.metrics.ssim
          })
          .catch((error) => {
            loadingModal.hide()
            showError("An error occurred: " + error.message)
          })
      })
    }
  
    // Handle extract form submission
    const extractForm = document.getElementById("extractForm")
    if (extractForm) {
      extractForm.addEventListener("submit", (e) => {
        e.preventDefault()
  
        const fileInput = document.getElementById("stegoImage")
  
        if (!fileInput.files[0]) {
          showError("Please select an image file.")
          return
        }
  
        // Show loading modal
        const loadingModalElement = document.getElementById("loadingModal")
        const loadingModal = new bootstrap.Modal(loadingModalElement)
        document.getElementById("loadingMessage").textContent = "Extracting message..."
        document.getElementById("loadingSubMessage").textContent = "This may take a few moments"
        loadingModal.show()
  
        // Create form data
        const formData = new FormData()
        formData.append("image", fileInput.files[0])
  
        // Send request to server
        fetch("/extract", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            loadingModal.hide()
  
            if (data.error) {
              showError(data.error)
              return
            }
  
            // Display extracted message
            document.getElementById("extractMessagePlaceholder").classList.add("d-none")
            document.getElementById("extractedMessageContainer").classList.remove("d-none")
            document.getElementById("extractedMessage").textContent = data.message
          })
          .catch((error) => {
            loadingModal.hide()
            showError("An error occurred: " + error.message)
          })
      })
    }
  
    // Display image preview when selected
    const coverImageInput = document.getElementById("coverImage")
    if (coverImageInput) {
      coverImageInput.addEventListener("change", function () {
        if (this.files && this.files[0]) {
          const reader = new FileReader()
  
          reader.onload = (e) => {
            const container = document.getElementById("originalImageContainer")
            container.innerHTML = `<img src="${e.target.result}" alt="Original Image">`
  
            // Show image details
            document.getElementById("imageDetails").classList.remove("d-none")
  
            // Create image object to get dimensions
            const img = new Image()
            img.onload = function () {
              document.getElementById("imageDimensions").textContent = `${this.width} × ${this.height}`
            }
            img.src = e.target.result
  
            // Get file details
            const file = coverImageInput.files[0]
            document.getElementById("imageFormat").textContent = file.type.split("/")[1].toUpperCase()
            document.getElementById("imageSize").textContent = `${(file.size / 1024).toFixed(1)} KB`
          }
  
          reader.readAsDataURL(this.files[0])
        }
      })
    }
  
    // Display stego image preview when selected
    const stegoImageInput = document.getElementById("stegoImage")
    if (stegoImageInput) {
      stegoImageInput.addEventListener("change", function () {
        if (this.files && this.files[0]) {
          const reader = new FileReader()
  
          reader.onload = (e) => {
            const container = document.getElementById("extractImageContainer")
            container.innerHTML = `<img src="${e.target.result}" alt="Stego Image">`
          }
  
          reader.readAsDataURL(this.files[0])
        }
      })
    }
  
    // Function to show error modal
    function showError(message) {
      const errorModalElement = document.getElementById("errorModal")
      const errorModal = new bootstrap.Modal(errorModalElement)
      document.getElementById("errorMessage").textContent = message
      errorModal.show()
    }
  })
  