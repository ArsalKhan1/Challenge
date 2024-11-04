// added by me

export async function fetchLLMResponse(prompt: string, model = "gpt-4", stream = true): Promise<string> {
    const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt, model, stream })
    });

    if (!response.ok) {
        throw new Error("Failed to fetch response from backend");
    }

    if (stream) {
        // Handle streaming response
        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        let result = "";

        if (reader) {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
            }
        }
        return result;
    } else {
        // For non-streaming response
        const result = await response.json();
        return result.content;
    }
}
