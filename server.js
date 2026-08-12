// server.js
// Universal backend proxy for OpenAI-compatible APIs.

const express = require("express");
const cors = require("cors");
const rateLimit = require("express-rate-limit");
const helmet = require("helmet");
const hpp = require("hpp");
const path = require("path");
const { URL } = require("url");
const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

const app = express();

const DEFAULT_API_BASE_URL = "https://api.openai.com/v1";

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "http://localhost:5173,http://localhost:3001,http://127.0.0.1:3001,http://127.0.0.1:5173")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
app.use(cors({ origin: ALLOWED_ORIGINS }));
app.use(express.json({ limit: "200kb" }));
app.use(hpp()); // Protect against HTTP Parameter Pollution attacks

// Serve the UI files (HTML, CSS, JS) from the project root
app.use(express.static(path.join(__dirname)));

app.set("trust proxy", 1);

// Add security headers using Helmet
// We disable CSP by default to ensure we don't accidentally break external frontend assets (e.g. CDNs)
app.use(helmet({
  contentSecurityPolicy: false,
}));

// Global rate limiting to prevent basic DoS/brute-force attacks across all endpoints
const globalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 150, // limit each IP to 150 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests from this IP, please try again after 15 minutes" }
});
app.use(globalLimiter);

const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 50, // Increased slightly to accommodate potential reconnections or smaller chunk tests
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests. Please wait a moment and try again." },
});

async function initDb() {
  await prisma.$connect();
}

app.get("/api/health", (req, res) => {
  res.json({ ok: true, defaultApiBaseUrl: DEFAULT_API_BASE_URL });
});

// Database API endpoints
app.get("/api/chats", async (req, res) => {
  try {
    const threads = await prisma.thread.findMany({
      orderBy: { updatedAt: 'desc' }
    });
    // Convert BigInt to number for JSON
    const serialized = threads.map(t => ({
      ...t,
      createdAt: Number(t.createdAt),
      updatedAt: Number(t.updatedAt)
    }));
    res.json(serialized);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch chats" });
  }
});

app.get("/api/chats/:id", async (req, res) => {
  try {
    const messages = await prisma.message.findMany({
      where: { threadId: req.params.id },
      orderBy: { id: 'asc' },
      select: { role: true, content: true, createdAt: true }
    });
    const serialized = messages.map(m => ({
      ...m,
      createdAt: Number(m.createdAt)
    }));
    res.json(serialized);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch messages" });
  }
});

app.delete("/api/chats/:id", async (req, res) => {
  try {
    await prisma.message.deleteMany({ where: { threadId: req.params.id } });
    await prisma.thread.delete({ where: { id: req.params.id } });
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete chat" });
  }
});

app.delete("/api/chats", async (req, res) => {
  try {
    await prisma.message.deleteMany();
    await prisma.thread.deleteMany();
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete all chats" });
  }
});


// Helper to detect local model server loopback URLs
function isLocalUrl(urlString) {
  try {
    const parsed = new URL(urlString);
    const host = parsed.hostname.toLowerCase();
    return (
      host === "localhost" ||
      host === "127.0.0.1" ||
      host === "::1" ||
      host === "0.0.0.0"
    );
  } catch (e) {
    return false;
  }
}

function isSafeUrl(urlString) {
  try {
    const parsed = new URL(urlString);
    const host = parsed.hostname.toLowerCase();

    // Allow local loopback explicitly for local model runners (Ollama, LM Studio, etc.)
    if (isLocalUrl(urlString)) {
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    }

    if (parsed.protocol !== "https:") return false;

    // Block private IP ranges for non-local URLs
    if (host.startsWith("10.")) return false;
    if (host.startsWith("192.168.")) return false;
    if (host.match(/^172\.(1[6-9]|2[0-9]|3[0-1])\./)) return false;
    if (host === "169.254.169.254") return false;
    if (host.endsWith(".local")) return false;

    return true;
  } catch (e) {
    return false;
  }
}

function isOllamaCloud(baseUrl) {
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host === "ollama.com" || host.endsWith(".ollama.com");
  } catch (e) {
    return false;
  }
}

function isOllamaNative(baseUrl) {
  if (isOllamaCloud(baseUrl)) return true;
  try {
    const parsed = new URL(baseUrl);
    return parsed.port === "11434" && !parsed.pathname.includes("/v1");
  } catch (e) {
    return false;
  }
}

// Models endpoint — expects { baseUrl, apiKey }
app.post("/api/models", async (req, res) => {
  try {
    let { baseUrl, apiKey } = req.body || {};

    if (!baseUrl) {
      baseUrl = process.env.OPENAI_BASE_URL || DEFAULT_API_BASE_URL;
    }
    if (!apiKey) {
      apiKey = process.env.OPENAI_API_KEY;
    }

    if (!baseUrl) return res.status(400).json({ error: "baseUrl is required" });
    if (!isSafeUrl(baseUrl)) return res.status(400).json({ error: "baseUrl is invalid or points to a restricted internal IP" });
    
    // For local servers, API key is optional (default to dummy key if empty)
    if (!apiKey && isLocalUrl(baseUrl)) {
      apiKey = "local-key";
    }
    if (!apiKey) return res.status(400).json({ error: "apiKey is required for cloud providers" });

    const url = isOllamaNative(baseUrl)
      ? baseUrl.replace(/\/$/, "") + "/api/tags"
      : baseUrl.replace(/\/$/, "") + "/models";
      
    const headers = {};
    if (apiKey && apiKey !== "local-key") {
      headers["Authorization"] = "Bearer " + apiKey;
    } else {
      headers["Authorization"] = "Bearer local-key";
    }

    const resp = await fetch(url, { method: "GET", headers });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      console.error("Upstream models error:", resp.status, JSON.stringify(data));
      return res.status(502).json({ error: "The AI provider could not list models. Check your server status or API key." });
    }

    let models = [];
    if (isOllamaNative(baseUrl)) {
      if (Array.isArray(data.models)) {
        models = data.models.map((m) => m && (m.name || m.model || m.id)).filter(Boolean);
      }
    } else if (Array.isArray(data.data)) {
      models = data.data.map((m) => m && (m.id || m.name)).filter(Boolean);
    } else if (Array.isArray(data.models)) {
      models = data.models.map((m) => (typeof m === "string" ? m : m && (m.id || m.name))).filter(Boolean);
    }
    models = Array.from(new Set(models)).sort();
    res.json({ models });
  } catch (err) {
    console.error("Models proxy error:", err.message);
    res.status(502).json({ error: "Could not reach the AI provider." });
  }
});

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return "messages array is required";
  for (const m of messages) {
    if (!m || (m.role !== "user" && m.role !== "assistant" && m.role !== "system")) {
      return "each message needs role: 'user', 'assistant', or 'system'";
    }
    if (typeof m.content !== "string" || m.content.length === 0) {
      return "each message needs non-empty string content";
    }
    if (m.content.length > 8000) return "message content too long (max 8000 chars)";
  }
  return null;
}

async function callOllamaNativeStream(baseUrl, apiKey, model, messages, res) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey && apiKey !== "local-key") headers["Authorization"] = "Bearer " + apiKey;

  const resp = await fetch(baseUrl.replace(/\/$/, "") + "/api/chat", {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
    }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error((data.error && (data.error.message || data.error)) || "Request failed");
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive"
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let assistantReply = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n').filter(l => l.trim() !== '');
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        const text = parsed.message?.content || "";
        assistantReply += text;
        if (text) res.write(`data: ${JSON.stringify({ text })}\n\n`);
      } catch (e) {}
    }
  }
  res.write(`data: [DONE]\n\n`);
  res.end();
  return assistantReply;
}

async function callOpenAIStream(baseUrl, apiKey, model, messages, res) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey && apiKey !== "local-key") headers["Authorization"] = "Bearer " + apiKey;

  const resp = await fetch(baseUrl.replace(/\/$/, "") + "/chat/completions", {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
      max_tokens: 2000,
    }),
  });
  
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error((data.error && data.error.message) || "Request failed");
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive"
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let assistantReply = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n').filter(l => l.trim() !== '');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const dataStr = line.slice(6).trim();
        if (dataStr === '[DONE]') continue;
        try {
          const parsed = JSON.parse(dataStr);
          const text = parsed.choices?.[0]?.delta?.content || "";
          assistantReply += text;
          if (text) res.write(`data: ${JSON.stringify({ text })}\n\n`);
        } catch (e) {}
      }
    }
  }
  res.write(`data: [DONE]\n\n`);
  res.end();
  return assistantReply;
}

// Chat endpoint — expects { baseUrl, apiKey, model, messages: [{ role, content }, ...], threadId, title }
app.post("/api/chat", chatLimiter, async (req, res) => {
  try {
    let { baseUrl, apiKey, model, messages, threadId, title } = req.body;

    if (!apiKey) {
      apiKey = process.env.OPENAI_API_KEY;
    }
    if (!model) {
      model = process.env.OPENAI_MODEL || "gpt-4o-mini";
    }

    if (!baseUrl) {
      baseUrl = process.env.OPENAI_BASE_URL || DEFAULT_API_BASE_URL;
    }

    if (!baseUrl) return res.status(400).json({ error: "baseUrl is required" });
    if (!isSafeUrl(baseUrl)) return res.status(400).json({ error: "baseUrl is invalid or points to a restricted internal IP" });
    
    if (!apiKey && isLocalUrl(baseUrl)) {
      apiKey = "local-key";
    }
    if (!apiKey) return res.status(400).json({ error: "apiKey is required for cloud providers" });
    if (!model) return res.status(400).json({ error: "model is required" });

    const messagesError = validateMessages(messages);
    if (messagesError) return res.status(400).json({ error: messagesError });

    // Save user state to DB before streaming
    if (threadId) {
      await prisma.thread.upsert({
        where: { id: threadId },
        update: { title: title || "New chat", updatedAt: Date.now() },
        create: { id: threadId, title: title || "New chat", createdAt: Date.now(), updatedAt: Date.now() }
      });
      
      await prisma.message.deleteMany({ where: { threadId } });
      await prisma.message.createMany({
        data: messages.map(m => ({
          threadId,
          role: m.role,
          content: m.content,
          createdAt: Date.now()
        }))
      });
    }

    let assistantReply = "";
    if (isOllamaNative(baseUrl)) {
      assistantReply = await callOllamaNativeStream(baseUrl, apiKey, model, messages, res);
    } else {
      assistantReply = await callOpenAIStream(baseUrl, apiKey, model, messages, res);
    }

    // Save assistant reply to DB after streaming completes
    if (threadId && assistantReply) {
      await prisma.message.create({
        data: { threadId, role: "assistant", content: assistantReply, createdAt: Date.now() }
      });
      await prisma.thread.update({
        where: { id: threadId },
        data: { updatedAt: Date.now() }
      });
    }

  } catch (err) {
    console.error("Upstream error:", err.message);
    if (!res.headersSent) {
      res.status(502).json({ error: err.message || "The AI provider could not fulfill the request." });
    } else {
      // If headers already sent, we must send an error down the stream
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    }
  }
});


app.get(/^(?!\/api\/).*/, (req, res) => {
  res.sendFile(path.join(__dirname, "chat-app-preview.html"));
});

// Generic Error Handler Middleware to prevent leaking server details to clients
app.use((err, req, res, next) => {
  console.error("Unhandled Server Error:", err.message || err);
  if (!res.headersSent) {
    res.status(500).json({ error: "An unexpected internal error occurred." });
  } else {
    res.end();
  }
});

const DEFAULT_PORT = parseInt(process.env.PORT, 10) || 3001;

function startServer(port) {
  initDb().then(() => {
    const server = app.listen(port, () => {
      console.log(`Backend listening on http://localhost:${port}`);
    });

    server.on("error", (err) => {
      if (err.code === "EADDRINUSE") {
        console.warn(`Port ${port} is in use. Trying port ${port + 1}...`);
        startServer(port + 1);
      } else {
        console.error("Server error:", err);
      }
    });
  }).catch(err => {
    console.error("Failed to initialize database", err);
    process.exit(1);
  });
}

startServer(DEFAULT_PORT);
