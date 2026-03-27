package mcp

import (
	"archon/internal/arch"
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Request defines the incoming JSON-RPC structure from VS Code
type Request struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Method  string `json:"method"`
	Params  Params `json:"params"`
}

type Params struct {
	File   string `json:"file"`
	Prompt string `json:"prompt"`
}

// Response defines the outgoing JSON-RPC structure back to VS Code
type Response struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// StartServer listens indefinitely on Stdin for IDE commands
func StartServer(agent *arch.Agent) {
	// We read line-by-line from Standard Input (sent by VS Code)
	scanner := bufio.NewScanner(os.Stdin)

	for scanner.Scan() {
		line := scanner.Text()

		var req Request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			sendError(0, "Invalid JSON-RPC format")
			continue
		}

		// Route the command
		switch req.Method {
		case "refactor":
			err := agent.Orchestrate(req.Params.File, req.Params.Prompt)
			if err != nil {
				sendError(req.ID, err.Error())
			} else {
				sendResult(req.ID, "Surgical edit applied successfully.")
			}
		default:
			sendError(req.ID, fmt.Sprintf("Method '%s' not supported", req.Method))
		}
	}
}

// sendResult writes a successful response back to Stdout
func sendResult(id int, result interface{}) {
	resp := Response{JSONRPC: "2.0", ID: id, Result: result}
	bytes, _ := json.Marshal(resp)
	fmt.Println(string(bytes))
}

// sendError writes a failure response back to Stdout
func sendError(id int, errMsg string) {
	resp := Response{JSONRPC: "2.0", ID: id, Error: errMsg}
	bytes, _ := json.Marshal(resp)
	fmt.Println(string(bytes))
}
