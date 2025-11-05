const chatbotIcon = document.getElementById("chatbot-icon");
const chatbotWindow = document.getElementById("chatbot-window");
const closeBtn = document.getElementById("close-btn");
const minimizeBtn = document.getElementById("minimize-btn");
const messagesDiv = document.getElementById("chat-box");
const textarea = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// Show chat window on chat icon click
chatbotIcon.addEventListener("click", () => {
  chatbotWindow.classList.remove("hidden");
  setTimeout(() => chatbotWindow.classList.add("show"), 10);
  chatbotIcon.style.display = "none";
});

// Close button (❌) -> reset conversation
closeBtn.addEventListener("click", async () => {
  chatbotWindow.classList.remove("show");
  setTimeout(async () => {
    chatbotWindow.classList.add("hidden");
    chatbotIcon.style.display = "flex";
    messagesDiv.innerHTML = ""; // clear chat messages
    try {
      await fetch("/reset", { method: "POST" });
    } catch (e) {
      console.log(e);
    }
  }, 400);
});

// Minimize button (▬) -> hide chat but keep messages
minimizeBtn.addEventListener("click", () => {
  chatbotWindow.classList.remove("show");
  setTimeout(() => {
    chatbotWindow.classList.add("hidden");
    chatbotIcon.style.display = "flex"; // floating icon visible
  }, 400);
});

// Textarea auto-resize and send button activation
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
  if (textarea.value.trim()) {
    sendBtn.classList.add("active");
    textarea.classList.add("typing");  // Add typing animation class
  } else {
    sendBtn.classList.remove("active");
    textarea.classList.remove("typing");
  }
});

// Send message on Enter
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Send message on button click
send.addEventListener("click", sendMessage);

// Send message function
async function sendMessage() {
  const userText = textarea.value.trim();
  if (!userText) return;

  // Add user message
  addMessage(userText, "user-msg");
  textarea.value = "";
  textarea.style.height = "auto";
  sendBtn.classList.remove("active");
  textarea.classList.remove("typing");

  // Add AI placeholder with typing dots
  const botBubble = addMessage("", "ai-msg");
  botBubble.innerHTML = `<div class="dot-typing"><span></span><span></span><span></span></div>`;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `question=${encodeURIComponent(userText)}`
    });
    const data = await res.json();
    const responseText = data.answer || "No response received.";
    
    // Replace typing dots with parsed markdown or plain text
    botBubble.innerHTML = typeof marked !== "undefined" ? marked.parse(responseText) : responseText;
    
    // Animate fade-in
    botBubble.style.opacity = 0;
    botBubble.style.transform = "translateY(10px)";
    setTimeout(() => {
      botBubble.style.transition = "opacity 0.3s, transform 0.3s";
      botBubble.style.opacity = 1;
      botBubble.style.transform = "translateY(0)";
    }, 50);

    // Add copy button for this AI message
    addCopyButton(botBubble);

  } catch (error) {
    botBubble.innerHTML = `🚨 Network error: ${error.message}`;
  }

  // Scroll to bottom
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Utility: add message to chat
function addMessage(text, className) {
  const msg = document.createElement("div");
  msg.className = className;

  if (className === "ai-msg") {
    // Wrap AI text in a paragraph for copying
    const para = document.createElement("p");
    para.innerHTML = text;
    msg.appendChild(para);
  } else {
    // For user messages, just plain text
    msg.textContent = text;
  }

  messagesDiv.appendChild(msg);
  setTimeout(() => {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }, 50);
  return msg;
}

// Add copy button to AI response message
function addCopyButton(aiMsgDiv) {
  // Check if copy button already exists
  if (aiMsgDiv.querySelector(".copy-btn")) return;

  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy";
  copyBtn.className = "copy-btn";
  copyBtn.title = "Copy response";

  aiMsgDiv.appendChild(copyBtn);

  copyBtn.addEventListener("click", () => {
    // Copy the AI response text (plain text only)
    const para = aiMsgDiv.querySelector("p");
    if (!para) return;
    const textToCopy = para.innerText || para.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 2000);
    }).catch(() => {
      copyBtn.textContent = "Failed";
    });
  });
}