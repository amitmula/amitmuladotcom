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
import { Agent, setGlobalDispatcher } from 'undici';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

const port = 3000;

// --- Configuration ---
const VECTOR_STORE_PATH = path.join(__dirname, "hnswlib_index");
const OLLAMA_BASE_URL = "http://ollama.home.arpa";
const EMBEDDING_MODEL = "nomic-embed-text";
const LLM_MODEL = "hf.co/LiquidAI/LFM2-1.2B-RAG-GGUF:Q4_K_M";
const dataDir = path.join(__dirname, "rag_text");

// Middleware
app.use(cors());
app.use(express.json());

// --- Set Global Undici Dispatcher for Connection Timeout ---
setGlobalDispatcher(
  new Agent({
    connect: {
      timeout: 60000 // 60-second connection timeout
    }
  })
);

// --- RAG Setup ---
let retriever;
let llm;

// --- Document Loading ---
async function loadDocs() {
  return await findTxtFiles(dataDir)
}

async function findTxtFiles(dir) {
  let results = [];

  // Read the current directory
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    console.log(`processing file -> ${filePath}`);
    const stats = fs.statSync(filePath);
    if (stats.isDirectory()) {
      // Recursively read subdirectories
      results = results.concat(findTxtFiles(filePath));
    } else if (file.endsWith('.txt') || file.endsWith('.md')) {
      // Add .txt file to the result
      results.push({
        pageContent: fs.readFileSync(filePath, 'utf8'),
        metadata: { source: filePath },
      });
    }
  }
  return results;
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
        baseUrl: OLLAMA_BASE_URL,
        timeout: 300000,
      }); vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
      console.log("Vector store loaded successfully.");
    } else {
      console.log("Vector store not found. Building and saving new index...");
      const docs = await loadDocs();

      if (docs.length === 0) {
        console.log("No documents found in 'data' directory. Skipping RAG initialization.");
        return;
      }

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1500,
        chunkOverlap: 200,
        separators: ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n\n", "\n", " ", ""],
      });

      const chunks = await splitter.splitDocuments(docs);

      vectorStore = await HNSWLib.fromDocuments(
        chunks,
        new OllamaEmbeddings({
          model: EMBEDDING_MODEL,
          baseUrl: OLLAMA_BASE_URL,
          timeout: 300000,
        }));
      await vectorStore.save(VECTOR_STORE_PATH);
      console.log("Index built and saved to disk. Startup time is now optimized!");
    }

    retriever = vectorStore.asRetriever({ k: 5 });

    llm = new Ollama({
      model: LLM_MODEL,
      baseUrl: OLLAMA_BASE_URL,
      temperature: 0.5,
      timeout: 300000,
    }); console.log(`✅ RAG Initialized Successfully using ollama server ${OLLAMA_BASE_URL} !`);

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
    } catch (error) {
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

    const fullPrompt = `
    You are an expert assistant with access to a special knowledge base.
    Answer the question truthfully based on the provided context.
    If the context doesn't contain the answer, say "I don't have enough information to answer that."
    Be concise and conversational. Avoid saying the sentence "Based on the provided context" as much as you can.
    
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

  } catch (error) {
    console.error('Error during RAG process:', error);
    res.status(500).send({ error: 'Failed to process RAG request.' });
  }
});


app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
  initializeRAG();
});
