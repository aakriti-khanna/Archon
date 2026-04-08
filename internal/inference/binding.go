package inference

/*
#cgo CFLAGS: -I./llama/include -I./llama/ggml/include -I./llama/common -I./llama
#cgo LDFLAGS: -L./llama/build/bin -L./llama/build/src -L./llama/build/ggml/src -L./llama/build/ggml/src/ggml-metal -L./llama/build/ggml/src/ggml-blas -L./llama/build -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas -lstdc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
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

func InitSystem() {
	C.llama_backend_init()
}

func FreeSystem() {
	C.llama_backend_free()
}

func LoadModel(modelPath string) (*LlamaEngine, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	mParams := C.llama_model_default_params()
	mParams.n_gpu_layers = C.int(99)

	model := C.llama_model_load_from_file(cModelPath, mParams)
	if model == nil {
		return nil, errors.New("failed to load model, check path or VRAM")
	}

	cParams := C.llama_context_default_params()
	cParams.n_ctx = C.uint32_t(8192) // 8K context window for full-file generation

	ctx := C.llama_init_from_model(model, cParams)
	if ctx == nil {
		C.llama_model_free(model)
		return nil, errors.New("failed to create inference context")
	}

	return &LlamaEngine{
		model:   model,
		context: ctx,
	}, nil
}

func (engine *LlamaEngine) Close() {
	if engine.context != nil {
		C.llama_free(engine.context)
	}
	if engine.model != nil {
		C.llama_model_free(engine.model)
	}
}

func (engine *LlamaEngine) Tokenize(text string, addBos bool) []C.llama_token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxTokens := C.int(len(text) + 1)
	tokens := make([]C.llama_token, maxTokens)

	nTokens := C.llama_tokenize(
		C.llama_model_get_vocab(engine.model),
		cText, C.int(len(text)),
		(*C.llama_token)(unsafe.Pointer(&tokens[0])),
		maxTokens, C.bool(addBos), C.bool(true),
	)

	if nTokens < 0 {
		return nil
	}
	return tokens[:nTokens]
}

func (engine *LlamaEngine) TokenToStr(token C.llama_token) string {
	buf := make([]byte, 32)
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

// Generate evaluates prompt and streams the response via callback
func (engine *LlamaEngine) Generate(
	prompt string,
	maxTokens int,
	streamCallback func(string),
) (string, error) {
	tokens := engine.Tokenize(prompt, true)
	if len(tokens) == 0 {
		return "", errors.New("failed to tokenize prompt")
	}

	allocSize := len(tokens) + maxTokens
	batch := C.llama_batch_init(C.int(allocSize), 0, 1)
	defer C.llama_batch_free(batch)

	bTokens := unsafe.Slice(batch.token, allocSize)
	bPos := unsafe.Slice(batch.pos, allocSize)
	bNSeqId := unsafe.Slice(batch.n_seq_id, allocSize)
	bSeqId := unsafe.Slice(batch.seq_id, allocSize)
	bLogits := unsafe.Slice(batch.logits, allocSize)

	for i, token := range tokens {
		bTokens[i] = token
		bPos[i] = C.llama_pos(i)
		bNSeqId[i] = C.int32_t(1)

		seqIds := unsafe.Slice(bSeqId[i], 1)
		seqIds[0] = C.int32_t(0)

		if i == len(tokens)-1 {
			bLogits[i] = C.int8_t(1)
		} else {
			bLogits[i] = C.int8_t(0)
		}
	}
	batch.n_tokens = C.int(len(tokens))

	if C.llama_decode(engine.context, batch) != 0 {
		return "", errors.New("llama_decode failed during prompt evaluation")
	}

	sampler := C.llama_sampler_init_greedy()
	defer C.llama_sampler_free(sampler)

	var result string
	currentPos := len(tokens)
	eosToken := C.llama_vocab_eos(C.llama_model_get_vocab(engine.model))

	for i := 0; i < maxTokens; i++ {
		nextToken := C.llama_sampler_sample(sampler, engine.context, batch.n_tokens-1)

		if nextToken == eosToken {
			break
		}

		tokenStr := engine.TokenToStr(nextToken)
		result += tokenStr

		// --- REAL-TIME STREAMING CALLBACK ---
		if streamCallback != nil {
			streamCallback(tokenStr)
		}

		bTokens[0] = nextToken
		bPos[0] = C.llama_pos(currentPos)
		bNSeqId[0] = C.int32_t(1)

		seqIds := unsafe.Slice(bSeqId[0], 1)
		seqIds[0] = C.int32_t(0)

		bLogits[0] = C.int8_t(1)
		batch.n_tokens = C.int(1)

		if C.llama_decode(engine.context, batch) != 0 {
			return result, errors.New("llama_decode failed during text generation")
		}
		currentPos++
	}

	return result, nil
}
