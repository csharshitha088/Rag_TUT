# Hybrid RAG-Based Document Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents using hybrid retrieval.

## Features

- PDF loading and automatic text chunking
- Semantic search using Chroma
- Keyword search using BM25
- Hybrid retrieval using EnsembleRetriever
- Local embeddings with `nomic-embed-text`
- Local answer generation using Phi-3 + Ollama

## Architecture

PDF → Chunking → Chroma + BM25 → Hybrid Retriever → Phi-3 → Answer

## Tech Stack

- Python
- LangChain
- Chroma
- BM25
- Ollama
- Phi-3
- nomic-embed-text

## Setup

```bash
pip install -r requirements.txt
```
Install and run Ollama, then:

ollama pull phi3
ollama pull nomic-embed-text

Run the notebook:
multi-modal.ipynb


Question:
What is the Transformer architecture?

The system retrieves relevant chunks from the PDF using both semantic and keyword search, then passes the retrieved context to Phi-3 to generate the answer.
