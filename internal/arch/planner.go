package arch

import (
	"archon/internal/inference"
	"archon/internal/tools/filesystem"
	"fmt"
	"os"
	"path/filepath"
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

	fmt.Fprintln(os.Stderr, "[System] Booting Llama.cpp Engine...")
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

// Orchestrate runs the core execution loop
func (a *Agent) Orchestrate(targetFile string, userPrompt string) error {
	fmt.Fprintf(os.Stderr, "[Agent] Received task: '%s' on %s\n", userPrompt, targetFile)

	// 1. ANALYSIS
	fmt.Fprintln(os.Stderr, "[Agent] Step 1: Reading active file context...")
	fileContent, err := a.FileTool.ReadFile(targetFile)
	if err != nil {
		fileContent = "" // If the file doesn't exist yet, don't fail, just proceed
	}

	// 2. PLANNING & GENERATION
	fmt.Fprintln(os.Stderr, "[Agent] Step 2: Querying LLM for code generation...")

	llmPrompt := fmt.Sprintf(`<|im_start|>system
You are Archon, a highly focused AI coding assistant.
The user is currently working in: %s

Your ONLY task is to fulfill the user's request.
You must output the target filename on the FIRST line, followed by the ENTIRE updated or new file content wrapped in a markdown code block.

If the user asks to create a new file, output that new filename. Otherwise, output the original filename.

STRICT FORMAT:
FILE: <filename>
`+"```"+`
<complete file code here>
`+"```"+`
<|im_end|>
<|im_start|>user
Current File Context:
%s

Request: %s<|im_end|>
<|im_start|>assistant
FILE: `, targetFile, fileContent, userPrompt)

	generatedCode, err := a.Engine.Generate(llmPrompt, 2048)
	if err != nil {
		return fmt.Errorf("LLM generation failed: %v", err)
	}

	if strings.TrimSpace(generatedCode) == "" {
		return fmt.Errorf("LLM failed to generate a response")
	}

	// Because we prefilled "FILE: " in the prompt, the output will start immediately with the filename
	fullOutput := "FILE: " + strings.TrimSpace(generatedCode)

	fmt.Fprintf(os.Stderr, "\n--- LLM RAW OUTPUT ---\n%s\n----------------------\n\n", fullOutput)

	// 3. PARSING
	filename, newFileContent, err := parseFileResponse(fullOutput)
	if err != nil {
		return err
	}

	// Resolve the absolute path for the target file dynamically
	dir := filepath.Dir(targetFile)
	finalPath := filepath.Join(dir, filepath.Base(filename))

	// 4. EXECUTION
	fmt.Fprintf(os.Stderr, "[Agent] Step 3: Writing content to %s...\n", finalPath)

	// Backup existing file if it is an overwrite (Archon Guard)
	if _, err := os.Stat(finalPath); err == nil {
		original, _ := os.ReadFile(finalPath)
		os.WriteFile(finalPath+".bak", original, 0644)
		fmt.Fprintf(os.Stderr, "[Archon Guard] Created backup at %s.bak\n", finalPath)
	} else {
		fmt.Fprintf(os.Stderr, "[Agent] Creating entirely new file: %s\n", finalPath)
	}

	// Write the new or overwritten file!
	err = os.WriteFile(finalPath, []byte(newFileContent), 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %v", err)
	}

	fmt.Fprintln(os.Stderr, "[Agent] Task completed successfully!")
	return nil
}

// parseFileResponse robustly extracts the filename and code block from the LLM output
func parseFileResponse(output string) (string, string, error) {
	lines := strings.SplitN(output, "\n", 2)
	if len(lines) < 2 {
		return "", "", fmt.Errorf("LLM output format invalid, could not find content")
	}

	// Parse the filename
	filename := strings.TrimSpace(strings.TrimPrefix(lines[0], "FILE:"))
	filename = strings.Trim(filename, "`*\"' ")

	if filename == "" {
		return "", "", fmt.Errorf("LLM failed to specify a filename")
	}

	content := lines[1]

	// Extract the code from the markdown block
	startMarker := "```"
	startIdx := strings.Index(content, startMarker)
	if startIdx != -1 {
		// skip the language identifier (e.g., ```javascript)
		nlIdx := strings.Index(content[startIdx:], "\n")
		if nlIdx != -1 {
			startContent := startIdx + nlIdx + 1
			endIdx := strings.LastIndex(content, "```")
			if endIdx > startContent {
				// FIXED: Returning all 3 arguments (filename, content, error)
				return filename, strings.TrimSpace(content[startContent:endIdx]), nil
			}
			// FIXED: Returning all 3 arguments
			return filename, strings.TrimSpace(content[startContent:]), nil
		}
	}

	// Absolute fallback
	// FIXED: Returning all 3 arguments
	return filename, strings.TrimSpace(content), nil
}
