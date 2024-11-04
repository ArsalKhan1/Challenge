// added by me

// v3:
// utils/api.ts

export async function getLLMResponse(
    message: string,
    model: string,
    onData: (data: string) => void
  ) {
    const response = await fetch("http://127.0.0.1:5000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        model,
        stream: true,  // Enable streaming
      }),
    });
  
    if (!response.body) {
      throw new Error("No response body");
    }
  
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let done = false;
  
    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      const chunkValue = decoder.decode(value);
      onData(chunkValue);
    }
  }
  



// v2: points to local backend
// export async function fetchLLMResponse(message: string, model: string, stream: boolean = false) {
//     const response = await fetch("http://127.0.0.1:5000/api/chat", {  // Backend URL
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ message, model, stream }),
//     });

//     if (!response.ok) {
//         throw new Error(`Error: ${response.statusText}`);
//     }
    
//     const data = await response.json();
//     return data.response;  // Handle response
// }



// v1
// export async function fetchLLMResponse(prompt: string, model = "gpt-4", stream = true): Promise<string> {
//     const response = await fetch("http://localhost:8000/generate", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json"
//         },
//         body: JSON.stringify({ prompt, model, stream })
//     });

//     if (!response.ok) {
//         throw new Error("Failed to fetch response from backend");
//     }

//     if (stream) {
//         // Handle streaming response
//         const reader = response.body?.getReader();
//         const decoder = new TextDecoder("utf-8");
//         let result = "";

//         if (reader) {
//             while (true) {
//                 const { done, value } = await reader.read();
//                 if (done) break;
//                 result += decoder.decode(value, { stream: true });
//             }
//         }
//         return result;
//     } else {
//         // For non-streaming response
//         const result = await response.json();
//         return result.content;
//     }
// }
