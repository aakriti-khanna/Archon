package arch

import (
	"archon/internal/inference" // ADDED
	"archon/internal/parser/treesitter"
	"archon/internal/tools/filesystem"
	"fmt"
	"strings"
)

// Agent represents our local coding assistant
type Agent struct {
	FileTool *filesystem.FileTool
	Engine   *inference.LlamaEngine // ADDED
}

// NewAgent initializes the FileTool and the Inference Engine
func NewAgent(projectRoot string, modelPath string) (*Agent, error) { // UPDATED signature
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

// Orchestrate runs the core execution loop: Analyze -> Plan -> Generate -> Edit
func (a *Agent) Orchestrate(targetFile string, userPrompt string) error {
	fmt.Printf("[Agent] Received task: '%s' on %s\n", userPrompt, targetFile)

	// 1. ANALYSIS (The Eyes)
	fmt.Println("[Agent] Step 1: Analyzing file structure...")
	blocks, err := treesitter.ExtractGoFunctions(targetFile)
	if err != nil {
		return fmt.Errorf("failed to analyze file: %v", err)
	}

	var targetBlock treesitter.CodeBlock
	for _, b := range blocks {
		if b.Name == "Start" {
			targetBlock = b
			break
		}
	}

	if targetBlock.Content == "" {
		return fmt.Errorf("could not find target function in file")
	}
	fmt.Printf("[Agent] Found target block: %s\n", targetBlock.Name)

	// 2. PLANNING & GENERATION (The Brain)
	fmt.Println("[Agent] Step 2: Querying LLM for refactor...")

	// Construct a strict system prompt to force the model to ONLY output code
	llmPrompt := fmt.Sprintf(`<|im_start|>system
You are Archon, an expert Go developer. You will be given a code block and a requested change. 
Return ONLY the modified code block. Do not include markdown formatting, explanations, or comments outside the code.<|im_end|>
<|im_start|>user
Original Code:
%s

Request: %s<|im_end|>
<|im_start|>assistant
`, targetBlock.Content, userPrompt)

	// Ask the model to generate the new code (max 256 tokens for this small edit)
	generatedCode, err := a.Engine.Generate(llmPrompt, 256)
	if err != nil {
		return fmt.Errorf("LLM generation failed: %v", err)
	}

	// Clean up the output (sometimes models add trailing spaces or markdown despite instructions)
	cleanCode := strings.TrimSpace(generatedCode)
	cleanCode = strings.TrimPrefix(cleanCode, "```go")
	cleanCode = strings.TrimSuffix(cleanCode, "```")

	fmt.Printf("\n--- LLM SUGGESTED EDIT ---\n%s\n--------------------------\n\n", cleanCode)

	// 3. EXECUTION (The Hands)
	fmt.Println("[Agent] Step 3: Applying surgical edit...")
	err = a.FileTool.SurgicalEdit(targetFile, targetBlock.Content, cleanCode)
	if err != nil {
		return fmt.Errorf("failed to apply edit: %v", err)
	}

	fmt.Println("[Agent] Task completed successfully!")
	return nil
}
