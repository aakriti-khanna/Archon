package filesystem

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type FileTool struct {
	ProjectRoot string
}

// NewFileTool initializes the tool and ensures the root path is absolute.
func NewFileTool(root string) (*FileTool, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	return &FileTool{ProjectRoot: absRoot}, nil
}

// validatePath is our Sandbox guardrail. It prevents directory traversal attacks (e.g., "../../etc/passwd").
func (ft *FileTool) validatePath(target string) (string, error) {
	// 1. Join the root and the target
	fullPath := filepath.Join(ft.ProjectRoot, target)

	// 2. Clean the path to resolve any ".." or "."
	cleanPath := filepath.Clean(fullPath)

	// 3. Ensure the resulting path still resides within the ProjectRoot
	rel, err := filepath.Rel(ft.ProjectRoot, cleanPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("security violation: path %s is outside project root", target)
	}

	return cleanPath, nil
}

// ReadFile safely reads a file's contents for the LLM context.
func (ft *FileTool) ReadFile(relPath string) (string, error) {
	safePath, err := ft.validatePath(relPath)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(safePath)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %v", err)
	}
	return string(content), nil
}

// backupFile creates a safe .bak copy of the specific file before Archon edits it.
func (ft *FileTool) backupFile(safePath string, content []byte) error {
	backupPath := safePath + ".bak"
	if err := os.WriteFile(backupPath, content, 0644); err != nil {
		return fmt.Errorf("failed to write backup file: %v", err)
	}
	return nil
}

// SurgicalEdit replaces a specific block of text with new text, safely backing up first.
func (ft *FileTool) SurgicalEdit(relPath, searchBlock, replaceBlock string) error {
	safePath, err := ft.validatePath(relPath)
	if err != nil {
		return err
	}

	// Read the current content
	contentBytes, err := os.ReadFile(safePath)
	if err != nil {
		return fmt.Errorf("failed to read file for editing: %v", err)
	}
	content := string(contentBytes)

	// Ensure the exact block exists before touching anything
	if !strings.Contains(content, searchBlock) {
		return fmt.Errorf("search block not found in %s; LLM might have hallucinated formatting", relPath)
	}

	// Safety First: Create a local .bak file instead of polluting git stash
	fmt.Printf("[Archon Guard] Creating .bak backup for %s...\n", relPath)
	if err := ft.backupFile(safePath, contentBytes); err != nil {
		return fmt.Errorf("aborting edit: failed to backup file: %v", err)
	}

	// Perform the replacement
	newContent := strings.Replace(content, searchBlock, replaceBlock, 1)

	// Write back atomically
	if err := os.WriteFile(safePath, []byte(newContent), 0644); err != nil {
		return fmt.Errorf("failed to write updated file: %v", err)
	}

	fmt.Printf("[Archon] Successfully applied surgical edit to %s\n", relPath)
	return nil
}
