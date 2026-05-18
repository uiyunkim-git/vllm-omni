package main

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Backend struct {
	Host           string `json:"host"`
	Port           int    `json:"port"`
	URL            *url.URL
	Proxy          *httputil.ReverseProxy
	ActiveRequests int32
	CachedModelID  string
}

type Config struct {
	AuthToken string                `json:"auth_token"`
	Models    map[string][]*Backend `json:"models"`
}

type ModelMetrics struct {
	activeRequests  int64
	completedCount  int64 // reset every second
	RPS             float64
	mu              sync.Mutex
}

var (
	configMutex  sync.RWMutex
	appConfig    Config
	routingMutex sync.Mutex

	metricsMu    sync.RWMutex
	modelMetrics = make(map[string]*ModelMetrics)
)

func getOrCreateMetrics(modelName string) *ModelMetrics {
	metricsMu.RLock()
	m, ok := modelMetrics[modelName]
	metricsMu.RUnlock()
	if ok {
		return m
	}
	metricsMu.Lock()
	if m, ok = modelMetrics[modelName]; !ok {
		m = &ModelMetrics{}
		modelMetrics[modelName] = m
	}
	metricsMu.Unlock()
	return m
}

func metricsLoop() {
	ticker := time.NewTicker(time.Second)
	for range ticker.C {
		metricsMu.RLock()
		for _, m := range modelMetrics {
			completed := atomic.SwapInt64(&m.completedCount, 0)
			m.mu.Lock()
			m.RPS = float64(completed)
			m.mu.Unlock()
		}
		metricsMu.RUnlock()
	}
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	metricsMu.RLock()
	out := make(map[string]map[string]interface{}, len(modelMetrics))
	for name, m := range modelMetrics {
		m.mu.Lock()
		rps := m.RPS
		m.mu.Unlock()
		out[name] = map[string]interface{}{
			"active_requests": atomic.LoadInt64(&m.activeRequests),
			"req_per_sec":     rps,
		}
	}
	metricsMu.RUnlock()

	json.NewEncoder(w).Encode(out)
}

func loadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var newConfig Config
	if err := json.Unmarshal(data, &newConfig); err != nil {
		return err
	}

	// Initialize URLs and Proxies
	for _, backends := range newConfig.Models {
		for _, b := range backends {
			targetURL, err := url.Parse(fmt.Sprintf("https://%s:%d", b.Host, b.Port))
			if err != nil {
				return err
			}
			b.URL = targetURL
			
			proxy := httputil.NewSingleHostReverseProxy(targetURL)
			
			// PERFORMANCE TUNING: Go ReverseProxy uses DefaultTransport which restricts 
			// MaxIdleConnsPerHost to just 2. Since our downstream CLI worker blasts
			// thousands of TCP connections, this bottleneck forces the proxy to drop them (502/500).
			// We MUST expand the proxy's own connection pool to the backends.
			// CRITICAL FIX: Uvicorn's default Keep-Alive timeout is 5 seconds!
			// If we set IdleConnTimeout > 5s, Go will try to reuse connections that Uvicorn
			// is already closing, resulting in random '500 Internal Server Error's under load over LAN.
			// Setting to 3 seconds preempts Uvicorn safely.
			// CRITICAL FIX: To definitively rule out Keep-Alive race conditions with
			// remote Uvicorn nodes dropping TCP connections unexpectedly, we will 
			// disable Keep-Alives entirely and force a fresh TCP socket per request.
			proxy.Transport = &http.Transport{
				MaxIdleConns:        4000,
				MaxIdleConnsPerHost: 1000,
				MaxConnsPerHost:     2000,
				IdleConnTimeout:     3 * time.Second,
				DisableKeepAlives:   true,
				TLSClientConfig:     &tls.Config{InsecureSkipVerify: true},
			}
			
			// Configure proxy error handler to track drops
			proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) {
				log.Printf("Proxy error to %s: %v", targetURL.String(), e)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadGateway)
				// Return JSON instead of default Go HTML 500/502 pages
				jsonErr := fmt.Sprintf(`{"error":{"message":"Upstream vLLM node %s failed: %v","type":"upstream_error","param":null,"code":"502"}}`, targetURL.String(), e)
				w.Write([]byte(jsonErr))
			}
			
			// Attach the stream filter interceptor for <think> tag removal
			proxy.ModifyResponse = streamFilterInterceptor
			
			b.Proxy = proxy
			
			// Asynchronously cache the backend's true Model ID for transparent rewriting
			go func(b *Backend) {
				client := &http.Client{
					Timeout: 5 * time.Second,
					Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}},
				}
				resp, err := client.Get(fmt.Sprintf("%s/v1/models", b.URL.String()))
				if err == nil {
					defer resp.Body.Close()
					var modelsResp struct {
						Data []struct {
							Id string `json:"id"`
						} `json:"data"`
					}
					if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err == nil && len(modelsResp.Data) > 0 {
						b.CachedModelID = modelsResp.Data[0].Id
					}
				}
			}(b)
		}
	}

	configMutex.Lock()
	appConfig = newConfig
	configMutex.Unlock()

	// Pre-create metrics entries for all configured models
	for modelName := range newConfig.Models {
		getOrCreateMetrics(modelName)
	}

	log.Println("Configuration reloaded successfully")
	return nil
}

func reloadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	err := loadConfig("/app/config.json")
	if err != nil {
		log.Printf("Failed to reload config: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// A captureResponseWriter intercepts responses without embedding HTTP components
// to avoid data races during concurrent proxy requests.
type captureResponseWriter struct {
	statusCode int
	body       *bytes.Buffer
	header     http.Header
}

func (w *captureResponseWriter) Header() http.Header { return w.header }
func (w *captureResponseWriter) WriteHeader(statusCode int) { w.statusCode = statusCode }
func (w *captureResponseWriter) Write(b []byte) (int, error) { return w.body.Write(b) }

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	configMutex.RLock()
	currentConfig := appConfig
	configMutex.RUnlock()

	// CORS Setup
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 1. Authenticate
	authHeader := r.Header.Get("Authorization")
	expectedAuth := "Bearer " + currentConfig.AuthToken
	if currentConfig.AuthToken != "" && authHeader != expectedAuth {
		http.Error(w, `{"error":{"message":"Authentication Error, Invalid API Key.","type":"auth_error","param":null,"code":"401"}}`, http.StatusUnauthorized)
		return
	}

	// 2. Extract model
	var requestedModel string
	var bodyBytes []byte

	if r.Method == http.MethodPost {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, `{"error":{"message":"Failed to read body"}}`, http.StatusBadRequest)
			return
		}
		r.Body.Close()

		var reqBody struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(bodyBytes, &reqBody); err == nil {
			// Often LiteLLM clients prepend openai/ to the model name, e.g. openai/gpt-oss-120b
			// vLLM doesn't usually care about provider prefixes, or it registers the exact string.
			// The caller will send whatever model name it wants. 
			requestedModel = reqBody.Model
		}
		
		// Restore body for proxy
		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		r.ContentLength = int64(len(bodyBytes))
	} else if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
		// Concurrently query one backend per configured model to gather metadata,
		// overriding the id with the proxy's configured served name.
		var wg sync.WaitGroup
		var mu sync.Mutex
		merged := make(map[string]json.RawMessage)

		for modelName, backends := range currentConfig.Models {
			if len(backends) == 0 {
				continue
			}

			wg.Add(1)
			go func(mName string, bList []*Backend) {
				defer wg.Done()

				// Need to handle panics safely
				defer func() {
					if err := recover(); err != nil {
						log.Printf("Recovered from panic during model aggregation for %s: %v", mName, err)
					}
				}()

				var validItem json.RawMessage

				// Try the available backends for this model
				for _, b := range bList {
					rec := &captureResponseWriter{
						statusCode: http.StatusOK,
						body:       &bytes.Buffer{},
						header:     make(http.Header),
					}

					rClone := r.Clone(r.Context())
					rClone.Header.Del("Origin")

					b.Proxy.ServeHTTP(rec, rClone)

					if rec.statusCode >= 200 && rec.statusCode < 300 {
						var resp struct {
							Data []json.RawMessage `json:"data"`
						}
						if err := json.Unmarshal(rec.body.Bytes(), &resp); err == nil && len(resp.Data) > 0 {
							var obj map[string]interface{}
							// Take the first returned model block from the backend as the template
							if err := json.Unmarshal(resp.Data[0], &obj); err == nil {
								obj["id"] = mName
								obj["root"] = mName
								if newItem, err := json.Marshal(obj); err == nil {
									validItem = newItem
									break
								}
							}
						}
					}
				}

				if validItem == nil {
					// Fallback to mock block if all backends for this model fail
					now := time.Now().Unix()
					mockStr := fmt.Sprintf(`{"id":"%s","object":"model","created":%d,"owned_by":"vllm","root":"%s","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-mock","object":"model_permission","created":%d,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}`, mName, now, mName, now)
					validItem = json.RawMessage(mockStr)
				}

				mu.Lock()
				merged[mName] = validItem
				mu.Unlock()
			}(modelName, backends)
		}

		wg.Wait()

		// Sort by ID for deterministic output
		var modelNames []string
		for name := range merged {
			modelNames = append(modelNames, name)
		}
		sort.Strings(modelNames)

		var finalData []json.RawMessage
		for _, name := range modelNames {
			finalData = append(finalData, merged[name])
		}

		if finalData == nil {
			finalData = []json.RawMessage{}
		}

		finalResp := map[string]interface{}{
			"object": "list",
			"data":   finalData,
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(finalResp)
		return
	}

	// Fallback alias resolution dynamically matching backend identities
	resolvedModelName := requestedModel
	backends, ok := currentConfig.Models[requestedModel]
	if !ok || len(backends) == 0 {
		for k, vList := range currentConfig.Models {
			if strings.EqualFold(k, requestedModel) || strings.EqualFold("openai/"+k, requestedModel) || strings.EqualFold("vllm/"+k, requestedModel) {
				backends = vList
				resolvedModelName = k
				ok = true
				break
			}
			// If requestedModel exactly matches the backend's actual model id
			for _, b := range vList {
				if b.CachedModelID != "" && strings.EqualFold(b.CachedModelID, requestedModel) {
					backends = vList
					resolvedModelName = k
					ok = true
					break
				}
			}
			if ok {
				break
			}
		}
	}

	if !ok || len(backends) == 0 {
		log.Printf("[Rejecting] Invalid model requested: %s", requestedModel)
		http.Error(w, fmt.Sprintf(`{"error":{"message":"Invalid model name passed in model=%s.","type":"invalid_request_error","param":null,"code":"400"}}`, requestedModel), http.StatusBadRequest)
		return
	}

	// 4. Retry loop (max 10 retries)
	maxRetries := 10
	var lastErrResponse *http.Response
	var lastErrBody []byte

	metrics := getOrCreateMetrics(resolvedModelName)
	atomic.AddInt64(&metrics.activeRequests, 1)
	defer atomic.AddInt64(&metrics.activeRequests, -1)

	for attempt := 0; attempt <= maxRetries; attempt++ {
		var bestBackend *Backend
		var minReqs int32 = -1

		routingMutex.Lock()
		for _, b := range backends {
			reqs := atomic.LoadInt32(&b.ActiveRequests)
			if minReqs == -1 || reqs < minReqs {
				minReqs = reqs
				bestBackend = b
			}
		}

		if bestBackend == nil {
			routingMutex.Unlock()
			log.Printf("[Rejecting] No active backends for %s", requestedModel)
			http.Error(w, `{"error":{"message":"No available backends"}}`, http.StatusServiceUnavailable)
			return
		}

		atomic.AddInt32(&bestBackend.ActiveRequests, 1)
		routingMutex.Unlock()

		log.Printf("[Routing] %s -> %s (Active: %d, Attempt: %d)", requestedModel, bestBackend.URL.String(), minReqs, attempt+1)

		// Create a custom response writer to catch 500s internally
		rec := &retryResponseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK, // default
			body:           &bytes.Buffer{},
			header:         make(http.Header),
		}

		// Reset request body and rewrite the model field if there's a mismatch
		if r.Method == http.MethodPost && bodyBytes != nil {
			if bestBackend.CachedModelID != "" && bestBackend.CachedModelID != requestedModel {
				var genericMap map[string]interface{}
				if err := json.Unmarshal(bodyBytes, &genericMap); err == nil {
					// Translate to the expected model ID
					genericMap["model"] = bestBackend.CachedModelID
					if newBody, err := json.Marshal(genericMap); err == nil {
						r.Body = io.NopCloser(bytes.NewReader(newBody))
						r.ContentLength = int64(len(newBody))
						r.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))
					} else {
						r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
					}
				} else {
					r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				}
			} else {
				r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			}
		}

		// Prevent downstream Ollama containers from enforcing their own CORS policies based on the Origin header,
		// since we handle CORS centrally here.
		r.Header.Del("Origin")

		bestBackend.Proxy.ServeHTTP(rec, r)
		
		atomic.AddInt32(&bestBackend.ActiveRequests, -1)
		log.Printf("[Completed] Route %s -> %s (Status: %d)", requestedModel, bestBackend.URL.String(), rec.statusCode)

		if rec.statusCode < 500 {
			atomic.AddInt64(&metrics.completedCount, 1)
		}

		// Check if we need to retry
		if rec.statusCode >= 500 {
			lastErrResponse = &http.Response{
				StatusCode: rec.statusCode,
				Header:     rec.header,
			}
			lastErrBody = rec.body.Bytes()
			
			log.Printf("🚨 BACKEND %d FROM %s (Attempt %d) 🚨", rec.statusCode, bestBackend.URL.Host, attempt+1)
			
			// If we have more retries, continue the loop
			if attempt < maxRetries {
				continue
			}
		}

		// Success or non-500 error, or we ran out of retries.
		// Write the recorded response to the actual client.
		for k, v := range rec.header {
			w.Header()[k] = v
		}
		w.WriteHeader(rec.statusCode)
		w.Write(rec.body.Bytes())
		return
	}

	// This should technically never be reached due to the return inside the loop,
	// but just in case we fall through, write the last error.
	if lastErrResponse != nil {
		for k, v := range lastErrResponse.Header {
			w.Header()[k] = v
		}
		w.WriteHeader(lastErrResponse.StatusCode)
		w.Write(lastErrBody)
	} else {
		http.Error(w, `{"error":{"message":"Max retries exceeded without a response"}}`, http.StatusBadGateway)
	}
}

// Custom ResponseWriter to intercept status code and body
type retryResponseWriter struct {
	http.ResponseWriter
	statusCode int
	body       *bytes.Buffer
	header     http.Header
}

func (w *retryResponseWriter) Header() http.Header {
	return w.header
}

func (w *retryResponseWriter) WriteHeader(statusCode int) {
	w.statusCode = statusCode
}

func (w *retryResponseWriter) Write(b []byte) (int, error) {
	return w.body.Write(b)
}

func main() {
	err := loadConfig("/app/config.json")
	if err != nil {
		log.Printf("Warning: initial config load failed: %v", err)
	}

	go metricsLoop()

	mux := http.NewServeMux()
	mux.HandleFunc("/reload", reloadHandler)
	mux.HandleFunc("/stats", statsHandler)
	mux.HandleFunc("/", proxyHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "4000"
	}

	log.Printf("Starting Custom vLLM Reverse Proxy on :%s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
