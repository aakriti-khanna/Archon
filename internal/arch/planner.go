package arch

import (
	"archon/internal/inference"
	"archon/internal/tools/filesystem"
	"errors"
	"fmt"
	"log"
	"strings"
)

// Agent represents our local coding assistant
type Agent struct {
	FileTool *filesystem.FileTool
	Engine   *inference.LlamaEngine
}

// NewAgent initializes the FileTool and the Inference Engine
func NewAgent(projectRoot string, modelPath string) (*Agent, error) {
	ft, err := filesystem.NewFileTool(projectRoot)
	if err != nil {
		return nil, err
	}

	fmt.Println("[System] Booting Llama.cpp Engine...")
	inference.InitSystem()
	engine, err := inference.LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %v", err)
	}

	return &Agent{
		FileTool: ft,
		Engine:   engine,
	}, nil
}

// Close gracefully shuts down the AI engine
func (a *Agent) Close() {
	if a.Engine != nil {
		a.Engine.Close()
	}
	inference.FreeSystem()
}

func (a *Agent) Orchestrate(targetFile string, userPrompt string) error {
	// Notice we use log.Printf instead of fmt.Printf!
	log.Printf("[Agent] Received task: '%s' on %s\n", userPrompt, targetFile)

	// 1. ANALYSIS
	log.Println("[Agent] Step 1: Reading entire file context...")
	fileContent, err := a.FileTool.ReadFile(targetFile)
	if err != nil {
		return fmt.Errorf("failed to read target file: %v", err)
	}

	// 2. PLANNING & GENERATION
	log.Println("[Agent] Step 2: Querying LLM for Search/Replace block...")
	llmPrompt := fmt.Sprintf(`<|im_start|>system
You are Archon, an expert AI coding assistant.
You will be given the entire current file content and a user request.
You must fulfill the user's request by outputting exactly ONE search and replace block.

RULES:
1. The SEARCH block MUST match the existing file content EXACTLY, character for character, including indentation.
2. The REPLACE block contains the new code.
3. Use the following strict format:

<<<<SEARCH
[exact old code to replace]
====
[new code to insert]
>>>>REPLACE<|im_end|>
<|im_start|>user
File Context:
%s

Request: %s<|im_end|>
<|im_start|>assistant
`, fileContent, userPrompt)

	generatedCode, err := a.Engine.Generate(llmPrompt, 1024)
	if err != nil {
		return fmt.Errorf("LLM generation failed: %v", err)
	}

	// Print raw output safely to Stderr via log
	log.Printf("\n--- LLM RAW OUTPUT ---\n%s\n----------------------\n\n", generatedCode)

	// 3. PARSING
	searchBlock, replaceBlock, err := parseSearchReplace(generatedCode)
	if err != nil {
		return fmt.Errorf("failed to parse SEARCH/REPLACE block: %v", err)
	}

	// 4. EXECUTION (Using the new universal tool!)
	log.Println("[Agent] Step 3: Applying universal surgical edit...")
	err = a.FileTool.SearchAndReplace(targetFile, searchBlock, replaceBlock)
	if err != nil {
		return fmt.Errorf("failed to apply edit: %v", err)
	}

	log.Println("[Agent] Task completed successfully!")
	return nil
}

// parseSearchReplace extracts the exact code from the LLM's formatted output
func parseSearchReplace(output string) (string, string, error) {
	searchMarker := "<<<<SEARCH"
	dividerMarker := "===="
	replaceMarker := ">>>>REPLACE"

	startIdx := strings.Index(output, searchMarker)
	divIdx := strings.Index(output, dividerMarker)
	endIdx := strings.Index(output, replaceMarker)

	if startIdx == -1 || divIdx == -1 || endIdx == -1 {
		return "", "", errors.New("LLM did not output proper SEARCH/REPLACE formatting markers")
	}

	// Extract the blocks and trim the leading/trailing newlines added by the formatting
	searchBlock := output[startIdx+len(searchMarker) : divIdx]
	replaceBlock := output[divIdx+len(dividerMarker) : endIdx]

	searchBlock = strings.TrimPrefix(searchBlock, "\n")
	searchBlock = strings.TrimSuffix(searchBlock, "\n")
	replaceBlock = strings.TrimPrefix(replaceBlock, "\n")
	replaceBlock = strings.TrimSuffix(replaceBlock, "\n")

	return searchBlock, replaceBlock, nil
}
