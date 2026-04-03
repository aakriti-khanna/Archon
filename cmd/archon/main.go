package main

import (
	"archon/internal/arch"
	"archon/pkg/mcp"
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	// Define the --serve flag
	serveMode := flag.Bool("serve", false, "Start Archon in JSON-RPC server mode for IDEs")
	flag.Parse()
	log.SetOutput(os.Stderr)
	// 1. Setup paths
	currentDir, _ := os.Getwd()
	modelPath := "./assets/models/qwen2.5-coder-7b.gguf"

	// 2. Initialize the Agent
	agent, err := arch.NewAgent(currentDir, modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize agent: %v", err)
	}
	defer agent.Close()

	// 3. Branch execution based on mode
	if *serveMode {
		// --- SERVER MODE ---
		// Notice we don't use fmt.Println here because Stdout is reserved for JSON communication
		os.Stderr.WriteString("[Archon] Started MCP Server. Listening on Stdin...\n")
		mcp.StartServer(agent)
	} else {
		// --- CLI MOCK MODE (For manual testing) ---
		fmt.Println("Starting Archon Execution Loop...")

		dummyGoFile := "dummy_server.go"
		os.WriteFile(dummyGoFile, []byte("package main\n\nfunc Start() error {\n\treturn nil\n}"), 0644)
		defer os.Remove(dummyGoFile)
		defer os.Remove(dummyGoFile + ".bak")

		err = agent.Orchestrate(dummyGoFile, "Add a fmt.Println stating 'Archon has taken control' at the beginning of the Start method.")
		if err != nil {
			log.Fatalf("Agent failed task: %v", err)
		}

		finalContent, _ := agent.FileTool.ReadFile(dummyGoFile)
		fmt.Printf("\n--- FINAL RESULT IN %s ---\n%s\n", dummyGoFile, finalContent)
	}
}
