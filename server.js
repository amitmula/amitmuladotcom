import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// LangChain imports
import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// --- RAG Setup ---
let retriever;
let llm;
const conversationHistory = [];

// --- Configuration ---
const VECTOR_STORE_PATH = path.join(__dirname, "hnswlib_index");
const OLLAMA_BASE_URL = "http://host.docker.internal:11434";
const EMBEDDING_MODEL = "nomic-embed-text";
const LLM_MODEL = "hf.co/LiquidAI/LFM2-1.2B-RAG-GGUF:Q4_K_M";

// --- Document Loading ---
async function loadDocs() {
  const dataDir = path.join(__dirname, "data");
  if (!fs.existsSync(dataDir)) {
    console.error(`Error: 'data' directory not found at ${dataDir}`);
    return [];
  }
  return fs
    .readdirSync(dataDir)
    .filter(f => f.endsWith(".txt"))
    .map(f => ({
      pageContent: fs.readFileSync(path.join(dataDir, f), "utf8"),
      metadata: { source: f },
    }));
}

async function initializeRAG() {
    console.log("Initializing RAG...");
    try {
        let vectorStore;

        console.log("Checking for existing vector store...");
        if (fs.existsSync(VECTOR_STORE_PATH)) {
            console.log("Loading vector store from disk...");
            const embeddings = new OllamaEmbeddings({
                model: EMBEDDING_MODEL,
                baseUrl: OLLAMA_BASE_URL 
            });
            vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
            console.log("Vector store loaded successfully.");
        } else {
            console.log("Vector store not found. Building and saving new index...");
            const docs = await loadDocs();

            if (docs.length === 0) {
                console.log("No documents found in 'data' directory. Skipping RAG initialization.");
                return;
            }

            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 800,
                chunkOverlap: 100,
            });

            const chunks = await splitter.splitDocuments(docs);

            vectorStore = await HNSWLib.fromDocuments(
                chunks,
                new OllamaEmbeddings({
                    model: EMBEDDING_MODEL,
                    baseUrl: OLLAMA_BASE_URL,
                })
            );
            await vectorStore.save(VECTOR_STORE_PATH);
            console.log("Index built and saved to disk. Startup time is now optimized!");
        }

        retriever = vectorStore.asRetriever({ k: 3 });

        llm = new Ollama({
            model: LLM_MODEL,
            baseUrl: OLLAMA_BASE_URL,
            temperature: 0.2,
        });
        console.log("✅ RAG Initialized Successfully!");

    } catch (error) {
        console.error("🚨 RAG Initialization Failed:", error);
    }
}


// --- API Endpoint ---
app.post('/api/chat', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt) {
        return res.status(400).send({ error: 'Prompt is required' });
    }

    // If RAG is not ready, fall back to a simple LLM call
    if (!retriever || !llm) {
        console.warn("RAG not initialized. Falling back to simple LLM response.");
        try {
            const stream = await llm.stream(prompt);
            res.setHeader('Content-Type', 'application/jsonl');
            for await (const chunk of stream) {
                 res.write(JSON.stringify({ response: chunk }) + '\n');
            }
            res.end();
        } catch(error) {
            console.error("Error during simple LLM fallback:", error);
            res.status(500).send({ error: "Failed to get response from LLM." });
        }
        return;
    }

    console.log(`RAG prompt received: "${prompt}"`);

    try {
        // Retrieve relevant documents
        const retrievedDocs = await retriever.getRelevantDocuments(prompt);
        const contextText = retrievedDocs.map(d => d.pageContent).join("\n\n");

        // For history, keep only the last 3 turns to further cut down prompt size.
        // NOTE: This global conversation history is shared across all clients.
        // For a production system, you would manage history per user/session.
        const historyText = conversationHistory
            .slice(-3) 
            .map((turn, i) => `Q${i + 1}: ${turn.question}\nA${i + 1}: ${turn.answer}`)
            .join("\n\n");

        const fullPrompt = `
You are a helpful assistant. Answer the question based ONLY on the context below.
Do NOT use outside knowledge. Be concise and conversational.
The context provided is authoritative.

${historyText ? `Previous conversation (last 3 turns):\n${historyText}\n\n` : ""}

Context:
${contextText}

Question:
${prompt}
`;
        
        res.setHeader('Content-Type', 'application/jsonl');
        let fullAnswer = "";

        const stream = await llm.stream(fullPrompt);
        for await (const chunk of stream) {
            const responseChunk = JSON.stringify({ response: chunk }) + '\n';
            res.write(responseChunk);
            fullAnswer += chunk;
        }
        res.end();

        // Save current turn to conversation history
        conversationHistory.push({ question: prompt, answer: fullAnswer });

    } catch (error) {
        console.error('Error during RAG process:', error);
        res.status(500).send({ error: 'Failed to process RAG request.' });
    }
});


// --- Server Startup ---
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
    // Initialize the RAG system after the server starts
    initializeRAG();
});
