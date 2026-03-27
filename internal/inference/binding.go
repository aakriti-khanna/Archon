package inference

/*
#cgo CFLAGS: -I./llama/include -I./llama/ggml/include -I./llama/common -I./llama
#cgo LDFLAGS: -L./llama/build/src -L./llama/build/ggml/src -L./llama/build/ggml/src/ggml-metal -L./llama/build/ggml/src/ggml-blas -L./llama/build -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas -lstdc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// LlamaEngine holds the pointers to the underlying C++ model and context
type LlamaEngine struct {
	model   *C.struct_llama_model
	context *C.struct_llama_context
}

// InitSystem initializes the backend (must be called once)
func InitSystem() {
	C.llama_backend_init()
}

// FreeSystem releases backend resources
func FreeSystem() {
	C.llama_backend_free()
}

// LoadModel loads a GGUF model from the given path
func LoadModel(modelPath string) (*LlamaEngine, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	// Get default model parameters
	mParams := C.llama_model_default_params()

	// UPDATED: Use the modern API for loading the model
	model := C.llama_model_load_from_file(cModelPath, mParams)
	if model == nil {
		return nil, errors.New("failed to load model, check path or VRAM")
	}

	// Get default context parameters
	cParams := C.llama_context_default_params()

	// UPDATED: Use the modern API for context initialization
	ctx := C.llama_init_from_model(model, cParams)
	if ctx == nil {
		C.llama_model_free(model) // UPDATED
		return nil, errors.New("failed to create inference context")
	}

	return &LlamaEngine{
		model:   model,
		context: ctx,
	}, nil
}

// Close safely frees the model and context
func (engine *LlamaEngine) Close() {
	if engine.context != nil {
		C.llama_free(engine.context)
	}
	if engine.model != nil {
		C.llama_model_free(engine.model) // UPDATED
	}
}

// ... (Keep your existing InitSystem, FreeSystem, LoadModel, and Close functions) ...

// Tokenize converts a Go string into a slice of Llama tokens
func (engine *LlamaEngine) Tokenize(text string, addBos bool) []C.llama_token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Allocate a buffer for the tokens (max length of the string is a safe bet)
	maxTokens := C.int(len(text) + 1)
	tokens := make([]C.llama_token, maxTokens)

	// Ask the C++ engine to tokenize the string
	nTokens := C.llama_tokenize(
		C.llama_model_get_vocab(engine.model),
		cText, C.int(len(text)),
		(*C.llama_token)(unsafe.Pointer(&tokens[0])),
		maxTokens, C.bool(addBos), C.bool(true),
	)

	if nTokens < 0 {
		return nil // Buffer was too small, but we allocated max string length so this is rare
	}

	return tokens[:nTokens]
}

// TokenToStr converts a single token back into a Go string
func (engine *LlamaEngine) TokenToStr(token C.llama_token) string {
	buf := make([]byte, 32) // 32 bytes is plenty for a single UTF-8 token
	size := C.llama_token_to_piece(
		C.llama_model_get_vocab(engine.model),
		token,
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int(len(buf)),
		0, C.bool(true),
	)
	if size < 0 {
		return ""
	}
	return string(buf[:size])
}

// Generate takes a prompt, evaluates it, and streams the response until EOS or maxTokens
// Generate takes a prompt, evaluates it, and streams the response until EOS or maxTokens
func (engine *LlamaEngine) Generate(prompt string, maxTokens int) (string, error) {
	// 1. Tokenize the input prompt
	tokens := engine.Tokenize(prompt, true)
	if len(tokens) == 0 {
		return "", errors.New("failed to tokenize prompt")
	}

	allocSize := len(tokens) + maxTokens

	// 2. Initialize a batch for evaluating the tokens
	batch := C.llama_batch_init(C.int(allocSize), 0, 1)
	defer C.llama_batch_free(batch)

	// In CGO, we cannot directly index C arrays. We must cast them to Go slices first.
	bTokens := unsafe.Slice(batch.token, allocSize)
	bPos := unsafe.Slice(batch.pos, allocSize)
	bNSeqId := unsafe.Slice(batch.n_seq_id, allocSize)
	bSeqId := unsafe.Slice(batch.seq_id, allocSize) // Array of pointers
	bLogits := unsafe.Slice(batch.logits, allocSize)

	// Add all prompt tokens to the batch
	for i, token := range tokens {
		bTokens[i] = token
		bPos[i] = C.llama_pos(i)
		bNSeqId[i] = C.int32_t(1)

		// Safely handle the double pointer for sequence ID
		seqIds := unsafe.Slice(bSeqId[i], 1)
		seqIds[0] = C.int32_t(0)

		// We only want to extract logits (probabilities) for the very last token
		if i == len(tokens)-1 {
			bLogits[i] = C.int8_t(1)
		} else {
			bLogits[i] = C.int8_t(0)
		}
	}
	batch.n_tokens = C.int(len(tokens))

	// 3. Evaluate the prompt tokens into the KV cache
	if C.llama_decode(engine.context, batch) != 0 {
		return "", errors.New("llama_decode failed during prompt evaluation")
	}

	// 4. Set up a simple Sampler
	sampler := C.llama_sampler_init_greedy()
	defer C.llama_sampler_free(sampler)

	// 5. The Generation Loop
	var result string
	currentPos := len(tokens)

	// UPDATED: Using the modern API for the EOS token
	eosToken := C.llama_vocab_eos(C.llama_model_get_vocab(engine.model))

	for i := 0; i < maxTokens; i++ {
		// Sample the next token based on the current context
		nextToken := C.llama_sampler_sample(sampler, engine.context, batch.n_tokens-1)

		// Check if the model is done talking
		if nextToken == eosToken {
			break
		}

		// Convert token to string and append to our result
		result += engine.TokenToStr(nextToken)

		// Prepare the batch for the next loop iteration (evaluate 1 token at a time)
		bTokens[0] = nextToken
		bPos[0] = C.llama_pos(currentPos)
		bNSeqId[0] = C.int32_t(1)

		seqIds := unsafe.Slice(bSeqId[0], 1)
		seqIds[0] = C.int32_t(0)

		bLogits[0] = C.int8_t(1)
		batch.n_tokens = C.int(1)

		// Evaluate this new single token
		if C.llama_decode(engine.context, batch) != 0 {
			return result, errors.New("llama_decode failed during text generation")
		}

		currentPos++
	}

	return result, nil
}
