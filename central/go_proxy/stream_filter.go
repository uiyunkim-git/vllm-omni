package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

func streamFilterInterceptor(res *http.Response) error {
	if res.StatusCode != http.StatusOK {
		return nil
	}

	contentType := res.Header.Get("Content-Type")
	if strings.Contains(contentType, "text/event-stream") {
		res.Body = NewThinkStreamReader(res.Body)
		res.Header.Del("Content-Length")
	} else if strings.Contains(contentType, "application/json") {
		bodyBytes, err := io.ReadAll(res.Body)
		if err == nil {
			var obj map[string]interface{}
			if err := json.Unmarshal(bodyBytes, &obj); err == nil {
				modified := false
				re := regexp.MustCompile(`(?s)<think>(.*?)(?:</think>\n*|$)`)
				
				// Try OpenAI format
				if choices, ok := obj["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if message, ok := choice["message"].(map[string]interface{}); ok {
							if content, ok := message["content"].(string); ok {
								message["content"] = re.ReplaceAllString(content, "\n_💭 Thinking..._\n$1\n\n")
								modified = true
							}
						}
					}
				}
				
				// Try Anthropic format
				if contentArr, ok := obj["content"].([]interface{}); ok && len(contentArr) > 0 {
					for _, blockRaw := range contentArr {
						if block, ok := blockRaw.(map[string]interface{}); ok {
							if t, _ := block["type"].(string); t == "text" {
								if text, ok := block["text"].(string); ok {
									block["text"] = re.ReplaceAllString(text, "\n_💭 Thinking..._\n$1\n\n")
									modified = true
								}
							}
						}
					}
				}
				
				if modified {
					if newBody, err := json.Marshal(obj); err == nil {
						res.Body = io.NopCloser(bytes.NewReader(newBody))
						res.ContentLength = int64(len(newBody))
						res.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))
						return nil
					}
				}
			}
			res.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
	}
	return nil
}

type thinkStreamReader struct {
	io.ReadCloser
	scanner     *bufio.Scanner
	thinkBuffer string
	inThink     bool
	nextBytes   []byte
}

func NewThinkStreamReader(rc io.ReadCloser) io.ReadCloser {
	return &thinkStreamReader{
		ReadCloser: rc,
		scanner:    bufio.NewScanner(rc),
	}
}

func (r *thinkStreamReader) Read(p []byte) (n int, err error) {
	if len(r.nextBytes) > 0 {
		n = copy(p, r.nextBytes)
		r.nextBytes = r.nextBytes[n:]
		return n, nil
	}

	for r.scanner.Scan() {
		line := r.scanner.Text()

		if line == "data: [DONE]" || line == "event: content_block_stop" {
			if !r.inThink && len(r.thinkBuffer) > 0 {
				flushStr := r.thinkBuffer
				r.thinkBuffer = ""
				var flushMsg string
				if line == "event: content_block_stop" {
					flushMsg = fmt.Sprintf("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":%q}}\n\n", flushStr)
				} else {
					flushMsg = fmt.Sprintf(`data: {"id":"chatcmpl-flush","choices":[{"delta":{"content":%q}}],"object":"chat.completion.chunk"}`+"\n", flushStr)
				}
				r.nextBytes = []byte(flushMsg + line + "\n")
			} else {
				r.nextBytes = []byte(line + "\n")
			}
			n = copy(p, r.nextBytes)
			r.nextBytes = r.nextBytes[n:]
			return n, nil
		}

		if !strings.HasPrefix(line, "data: ") {
			r.nextBytes = []byte(line + "\n")
			n = copy(p, r.nextBytes)
			r.nextBytes = r.nextBytes[n:]
			return n, nil
		}

		jsonStr := strings.TrimPrefix(line, "data: ")
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err == nil {
			var contentStr string
			var contentFound bool
			var isAnthropic bool

			// Try OpenAI format
			if choices, ok := obj["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if c, ok := delta["content"].(string); ok {
							contentStr = c
							contentFound = true
						}
					}
				}
			}

			// Try Anthropic format if OpenAI failed
			if !contentFound {
				if t, ok := obj["type"].(string); ok && t == "content_block_delta" {
					if delta, ok := obj["delta"].(map[string]interface{}); ok {
						if text, ok := delta["text"].(string); ok {
							contentStr = text
							contentFound = true
							isAnthropic = true
						}
					}
				}
			}

			if contentFound {
				r.thinkBuffer += contentStr
				emittedContent := ""

				for {
					if !r.inThink {
						idx := strings.Index(r.thinkBuffer, "<think>")
						if idx != -1 {
							emittedContent += r.thinkBuffer[:idx] + "\n_💭 Thinking..._\n"
							r.thinkBuffer = r.thinkBuffer[idx+7:]
							r.inThink = true
							continue
						} else {
							runes := []rune(r.thinkBuffer)
							if len(runes) > 8 {
								split := len(runes) - 8
								emittedContent += string(runes[:split])
								r.thinkBuffer = string(runes[split:])
							}
							break
						}
					} else {
						idx := strings.Index(r.thinkBuffer, "</think>")
						if idx != -1 {
							emittedContent += r.thinkBuffer[:idx] + "\n\n"
							r.thinkBuffer = r.thinkBuffer[idx+8:]
							for strings.HasPrefix(r.thinkBuffer, "\n") {
								r.thinkBuffer = r.thinkBuffer[1:]
							}
							r.inThink = false
							continue
						} else {
							runes := []rune(r.thinkBuffer)
							if len(runes) > 8 {
								split := len(runes) - 8
								// We now EMIT the thinking progress instead of dropping it
								emittedContent += string(runes[:split])
								r.thinkBuffer = string(runes[split:])
							}
							break
						}
					}
				}

				// We must not `continue` and drop the chunk if emittedContent == ""
				// because the SSE stream already sent the `event: ` header for this chunk.
				// Dropping the `data: ` line causes malformed SSE frames and client JSON parse crashes.

				if isAnthropic {
					if delta, ok := obj["delta"].(map[string]interface{}); ok {
						delta["text"] = emittedContent
						newJson, _ := json.Marshal(obj)
						r.nextBytes = []byte("data: " + string(newJson) + "\n")
						n = copy(p, r.nextBytes)
						r.nextBytes = r.nextBytes[n:]
						return n, nil
					}
				} else {
					if choices, ok := obj["choices"].([]interface{}); ok && len(choices) > 0 {
						if choice, ok := choices[0].(map[string]interface{}); ok {
							if delta, ok := choice["delta"].(map[string]interface{}); ok {
								delta["content"] = emittedContent
								newJson, _ := json.Marshal(obj)
								r.nextBytes = []byte("data: " + string(newJson) + "\n")
								n = copy(p, r.nextBytes)
								r.nextBytes = r.nextBytes[n:]
								return n, nil
							}
						}
					}
				}
			}
		}

		r.nextBytes = []byte(line + "\n")
		n = copy(p, r.nextBytes)
		r.nextBytes = r.nextBytes[n:]
		return n, nil
	}

	if err := r.scanner.Err(); err != nil {
		return 0, err
	}

	return 0, io.EOF
}
