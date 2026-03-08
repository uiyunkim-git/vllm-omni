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
}

type Config struct {
	AuthToken string                `json:"auth_token"`
	Models    map[string][]*Backend `json:"models"`
}

var (
	configMutex sync.RWMutex
	appConfig   Config
	routingMutex sync.Mutex
)

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
			// Prevent duplicate CORS headers (we already set them in proxyHandler)
			proxy.ModifyResponse = func(resp *http.Response) error {
				resp.Header.Del("Access-Control-Allow-Origin")
				resp.Header.Del("Access-Control-Allow-Methods")
				resp.Header.Del("Access-Control-Allow-Headers")

				// Deep Debugging: who is sending the 500 error?
				if resp.StatusCode == 500 || resp.StatusCode == 502 {
					log.Printf("🚨 BACKEND 500 FROM %s 🚨", resp.Request.URL.Host)
					for k, v := range resp.Header {
						log.Printf("Header: %s: %s", k, v)
					}
					// Dump a bit of the body
					if resp.Body != nil {
						bodyBytes, _ := io.ReadAll(resp.Body)
						resp.Body.Close()

						// We need a helper min function
						dumpLen := len(bodyBytes)
						if dumpLen > 100 {
							dumpLen = 100
						}
						log.Printf("Body snippet: %s", string(bodyBytes[:dumpLen]))
						// Restore body
						resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
					}
				}
				return nil
			}
			b.Proxy = proxy
		}
	}

	configMutex.Lock()
	appConfig = newConfig
	configMutex.Unlock()

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
		// Mock models list or pass to random node
		for _, backends := range currentConfig.Models {
			if len(backends) > 0 {
				backends[0].Proxy.ServeHTTP(w, r)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[]}`))
		return
	}

	var bestBackend *Backend
	var minReqs int32 = -1

	// Fallback alias resolution: "gpt-oss-120b" vs "openai/gpt-oss-120b"
	backends, ok := currentConfig.Models[requestedModel]
	if !ok || len(backends) == 0 {
		if len(requestedModel) > 7 && requestedModel[:7] == "openai/" {
			stripped := requestedModel[7:]
			backends, ok = currentConfig.Models[stripped]
		} else if len(requestedModel) > 0 {
			prefixed := "openai/" + requestedModel
			backends, ok = currentConfig.Models[prefixed]
		}
	}

	if !ok || len(backends) == 0 {
		log.Printf("[Rejecting] Invalid model requested: %s", requestedModel)
		http.Error(w, fmt.Sprintf(`{"error":{"message":"Invalid model name passed in model=%s.","type":"invalid_request_error","param":null,"code":"400"}}`, requestedModel), http.StatusBadRequest)
		return
	}

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

	// 4. Forward
	atomic.AddInt32(&bestBackend.ActiveRequests, 1)
	routingMutex.Unlock()
	
	defer atomic.AddInt32(&bestBackend.ActiveRequests, -1)

	log.Printf("[Routing] %s -> %s (Active: %d)", requestedModel, bestBackend.URL.String(), minReqs)

	bestBackend.Proxy.ServeHTTP(w, r)
	log.Printf("[Completed] Route %s -> %s", requestedModel, bestBackend.URL.String())
}

func main() {
	err := loadConfig("/app/config.json")
	if err != nil {
		log.Printf("Warning: initial config load failed: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/reload", reloadHandler)
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
