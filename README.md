# Archon: Local AI Coding Agent

Archon is a high-performance, local-first AI coding agent that enables developers to edit files, refactor code, and receive architectural suggestions directly within their local environment. 

By wrapping a C++ inference engine (`llama.cpp`) with Golang, Archon ensures zero-latency local operation, complete privacy, and deep file system integration while relying on AST (Tree-sitter) for surgical code understanding.

## Core Architecture

Archon is designed with a three-layer architecture:
1. **Engine Layer (C++)**: Handles GGUF model execution and KV cache management using `llama.cpp` for optimal performance on Apple Silicon (Metal) and Nvidia GPUs.
2. **Logic Layer (Go Wrapper)**: The "Brain" of the operation. It manages task planning, file interaction constraints, CGO bindings, and Tree-sitter AST parsing.
3. **Interface Layer (LSP/API)**: Exposes the agent's capabilities via JSON-RPC/gRPC and the Model Context Protocol (MCP) to IDE integrations (e.g., VS Code, JetBrains).

## Key Features

- **Surgical File Editing**: Avoids full-file rewrites by identifying target blocks and applying precise Search/Replace changes natively.
- **Fail-safe Operations**: Automatically stashes work via Git before executing changes.
- **Graph-based RAG**: Understands repository context by following AST import trees, selectively injecting only relevant snippets to conserve token limits.
- **Execution & Validation Loop**: Automatically runs `go vet`, `npm test`, or syntax checkers after edits to catch errors and self-correct prior to developer review.

## Resource Requirements

**Minimum Specs:**
- MacBook Pro (M1/M2/M3) or PC with 16GB+ RAM.
- GPU with Metal (macOS) or CUDA (Windows/Linux) support.
- Recommended Models: `Qwen2.5-Coder-7B` (Standard) or `DeepSeek-Coder-32B` (Complex tasks).

## Security & Governance

- **100% Local**: No proprietary code ever leaves your machine.
- **Sandboxed**: File modifications are securely restricted to the project root.
- **Human-in-the-Loop**: All write actions act as proposals and require a final Y/N physical confirmation.
