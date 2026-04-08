package arch

import (
	"archon/internal/inference"
	"archon/internal/tools/filesystem"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Agent struct {
	FileTool *filesystem.FileTool
	Engine   *inference.LlamaEngine
}

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

func (a *Agent) Close() {
	if a.Engine != nil {
		a.Engine.Close()
	}
	inference.FreeSystem()
}

// Note the updated return signature: (string, string, error)
func (a *Agent) Orchestrate(targetFile string, userPrompt string) (string, string, error) {
	fmt.Fprintf(os.Stderr, "[Agent] Received task: '%s' on %s\n", userPrompt, targetFile)

	// 1. ANALYSIS
	fmt.Fprintln(os.Stderr, "[Agent] Step 1: Reading active file context...")
	fileContent, err := a.FileTool.ReadFile(targetFile)
	if err != nil {
		fileContent = "" // Proceed even if the file is completely empty
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

	// --- REAL-TIME STREAMING SETUP ---
	streamCb := func(token string) {
		msg := map[string]interface{}{
			"jsonrpc": "2.0",
			"method":  "stream",
			"params": map[string]string{
				"token": token,
			},
		}
		b, _ := json.Marshal(msg)
		fmt.Println(string(b))
	}

	generatedCode, err := a.Engine.Generate(llmPrompt, 2048, streamCb)
	if err != nil {
		return "", "", fmt.Errorf("LLM generation failed: %v", err)
	}

	if strings.TrimSpace(generatedCode) == "" {
		return "", "", fmt.Errorf("LLM failed to generate a response")
	}

	fullOutput := "FILE: " + strings.TrimSpace(generatedCode)
	fmt.Fprintf(os.Stderr, "\n--- LLM RAW OUTPUT ---\n%s\n----------------------\n\n", fullOutput)

	// 3. PARSING
	filename, newFileContent, err := parseFileResponse(fullOutput)
	if err != nil {
		return "", "", err
	}

	// Resolve the absolute path
	dir := filepath.Dir(targetFile)
	finalPath := filepath.Join(dir, filepath.Base(filename))

	// 4. EXECUTION
	fmt.Fprintf(os.Stderr, "[Agent] Step 3: Sending proposal back to IDE for Diff...\n")

	// WE NO LONGER WRITE TO THE DISK HERE!
	// We return the path and code to the server so VS Code can handle the Diff view.
	return finalPath, newFileContent, nil
}

func parseFileResponse(output string) (string, string, error) {
	lines := strings.SplitN(output, "\n", 2)
	if len(lines) < 2 {
		return "", "", fmt.Errorf("LLM output format invalid, could not find content")
	}

	filename := strings.TrimSpace(strings.TrimPrefix(lines[0], "FILE:"))
	filename = strings.Trim(filename, "`*\"' ")

	if filename == "" {
		return "", "", fmt.Errorf("LLM failed to specify a filename")
	}

	content := lines[1]
	startMarker := "```"
	startIdx := strings.Index(content, startMarker)
	if startIdx != -1 {
		nlIdx := strings.Index(content[startIdx:], "\n")
		if nlIdx != -1 {
			startContent := startIdx + nlIdx + 1
			endIdx := strings.LastIndex(content, "```")
			if endIdx > startContent {
				return filename, strings.TrimSpace(content[startContent:endIdx]), nil
			}
			return filename, strings.TrimSpace(content[startContent:]), nil
		}
	}

	return filename, strings.TrimSpace(content), nil
}
