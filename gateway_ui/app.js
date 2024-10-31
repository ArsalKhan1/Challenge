document.getElementById("chat-form").addEventListener("submit", async function (event) {
    event.preventDefault();

    const userInput = document.getElementById("user-input").value;
    const responseContainer = document.getElementById("response-container");
    responseContainer.innerHTML = "";  // Clear previous response

    // Make the request to the backend
    const eventSource = new EventSourcePolyfill('/generate', {
        headers: { "Content-Type": "application/json" },
        method: "POST",
        body: JSON.stringify({ message: userInput })
    });

    eventSource.onmessage = function (event) {
        const data = event.data;

        // Append each chunk to the response container
        responseContainer.innerHTML += data;
    };

    eventSource.onerror = function (error) {
        console.error("Error:", error);
        eventSource.close();
    };
});
