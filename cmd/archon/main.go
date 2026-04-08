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
	serveMode := flag.Bool("serve", false, "Start Archon in JSON-RPC server mode for IDEs")
	flag.Parse()

	currentDir, _ := os.Getwd()
	modelPath := "./assets/models/qwen2.5-coder-7b.gguf"

	agent, err := arch.NewAgent(currentDir, modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize agent: %v", err)
	}
	defer agent.Close()

	if *serveMode {
		os.Stderr.WriteString("[Archon] Started MCP Server. Listening on Stdin...\n")
		mcp.StartServer(agent)
	} else {
		fmt.Println("Starting Archon Execution Loop...")
		dummyGoFile := "dummy_server.go"
		os.WriteFile(dummyGoFile, []byte("package main\n\nfunc Start() error {\n\treturn nil\n}"), 0644)
		defer os.Remove(dummyGoFile)
		defer os.Remove(dummyGoFile + ".bak")

		// FIXED: Capture all 3 return values from Orchestrate
		targetPath, newContent, err := agent.Orchestrate(dummyGoFile, "Add a fmt.Println stating 'Archon has taken control'")
		if err != nil {
			log.Fatalf("Agent failed task: %v", err)
		}

		// Since it no longer writes to disk automatically, we just print the result for manual testing
		fmt.Printf("\n--- PROPOSED FILE: %s ---\n%s\n", targetPath, newContent)
	}
}
