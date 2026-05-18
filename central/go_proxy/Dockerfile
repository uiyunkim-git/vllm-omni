FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod ./
COPY *.go ./

RUN go build -o vllm_router .

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/vllm_router .

EXPOSE 4000
CMD ["./vllm_router"]
