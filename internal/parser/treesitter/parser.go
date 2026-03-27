package treesitter

import (
	"context"
	"fmt"
	"os"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
)

// CodeBlock represents a perfectly extracted chunk of code (like a struct or func)
type CodeBlock struct {
	Name    string
	Content string
}

// ExtractGoFunctions parses a file and extracts all function and method definitions.
func ExtractGoFunctions(filePath string) ([]CodeBlock, error) {
	// Read the source code
	sourceCode, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	// Initialize the parser with Go grammar
	parser := sitter.NewParser()
	parser.SetLanguage(golang.GetLanguage())

	// Parse the code into an AST
	tree, err := parser.ParseCtx(context.Background(), nil, sourceCode)
	if err != nil {
		return nil, fmt.Errorf("failed to parse AST: %v", err)
	}

	var blocks []CodeBlock

	// We use a Tree-sitter Query to find all function declarations
	// This query looks for standard functions and methods
	queryStr := `
	(function_declaration
		name: (identifier) @func.name) @func.def
	
	(method_declaration
		name: (field_identifier) @method.name) @method.def
	`

	q, err := sitter.NewQuery([]byte(queryStr), golang.GetLanguage())
	if err != nil {
		return nil, fmt.Errorf("failed to compile query: %v", err)
	}

	qc := sitter.NewQueryCursor()
	qc.Exec(q, tree.RootNode())

	for {
		m, ok := qc.NextMatch()
		if !ok {
			break
		}

		// Iterate through the captured nodes in the match
		var name, content string
		for _, c := range m.Captures {
			captureName := q.CaptureNameForId(c.Index)
			nodeContent := c.Node.Content(sourceCode)

			if captureName == "func.name" || captureName == "method.name" {
				name = nodeContent
			}
			if captureName == "func.def" || captureName == "method.def" {
				content = nodeContent
			}
		}

		if name != "" && content != "" {
			blocks = append(blocks, CodeBlock{
				Name:    name,
				Content: content,
			})
		}
	}

	return blocks, nil
}
