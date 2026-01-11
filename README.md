# Built-an-On-Premise-University-Knowledge-Based-AI-Agent-using-LLMs-GPU-Acceleration
On-prem university knowledge-based AI agent using LLMs, GPU-accelerated OCR, vector search and RAG. Converts large PDFs (500–600 pages), builds a semantic knowledge base with ChromaDB, and answers queries using local Mistral via Ollama. Fully private, offline deployment on RTX 4070 Ti (12GB).
Over the past few weeks, I designed and implemented a complete end-to-end Knowledge-Based AI Agent for a university environment - running entirely on my own local GPU infrastructure.

🧠 What the system does

This AI assistant can:

Ingest large academic & administrative PDFs (500–600 pages)

Automatically detect scanned documents and apply OCR

Convert everything into clean structured text

Build a high-performance vector knowledge base

Answer user queries strictly from the university’s internal documents

Provide source citations for every response

🏗 Architecture Overview

1️⃣ Intelligent Document Ingestion

PyMuPDF for text extraction

EasyOCR with GPU acceleration

Automatic fallback to OCR when normal text extraction fails

Text cleaning & normalization pipeline

2️⃣ Knowledge Base Construction

Chunking with RecursiveCharacterTextSplitter

Embeddings using HuggingFace MiniLM

Vector storage using ChromaDB

3️⃣ Retrieval-Augmented Generation

Local Mistral LLM via Ollama

RetrievalQA chain from LangChain

Answers generated only from institutional knowledge

4️⃣ Production-Ready UI

Built with Streamlit

Clean chat interface

Source document transparency

Fully offline / on-prem deployment

🖥 Infrastructure

GPU: NVIDIA RTX 4070 Ti

VRAM: 12GB

Fully local, private deployment

No cloud dependency

💻 Tech Stack

Python | PyMuPDF | EasyOCR | OpenCV | PyTorch | LangChain | ChromaDB | HuggingFace | Ollama | Mistral | Streamlit | GPU

🌍 Why this matters

This system demonstrates how institutions can deploy secure, private, high-performance AI assistants without sending data to the cloud — ensuring:

Data privacy

Cost control

Full infrastructure ownership

This project blends LLM engineering, document intelligence, vector search, and GPU acceleration into a real production-grade solution.
