# 단일 진입점(게이트웨이) · 인증 · 도메인

## 왜 앞에 프록시가 필요한가 (그리고 왜 느려지지 않는가)

웹(central :8080)과 추론 API(Dynamo 프론트 :11434)는 **서로 다른 프로그램**이다.
하나의 도메인·하나의 포트로 둘 다 서비스하려면 경로로 갈라줄 무언가가 반드시 앞에 있어야 한다.

검토한 대안과 결론:

| 안 | 가능? | 판단 |
|---|---|---|
| 웹을 Dynamo(Rust) 뒤에 숨긴다 | ❌ | `dynamo.frontend`는 추론 서버다. 정적 파일 서빙·리버스 프록시 기능이 없고 설정 항목도 없다 |
| central(FastAPI)이 `/v1`도 대신 받는다 | ❌ | Python 홉을 LLM 경로에 다시 넣는 것 — 이 플랫폼이 걷어낸 바로 그 병목 |
| 301/307 리다이렉트로 `:11434`로 넘긴다 | ❌ | 클라이언트가 그 포트에 직접 닿아야 하므로 "한 포트"가 깨지고 왕복이 늘어난다 |
| **얇은 프록시(Caddy)를 앞에 둔다** | ✅ | 아래 실측대로 오버헤드가 측정 한계 이하 |

### 실측 (gpt-oss-120b, believe 워크로드, 같은 시점 연속 측정)

| 부하 | 직접 `:11434` | 게이트웨이 `:8443` |
|---|---|---|
| 동시성 8 (오버헤드가 가장 잘 보이는 구간) | 8.73 req/s · p50 **0.79s** | 9.15 req/s · p50 **0.78s** |
| 동시성 512 | 101.3 req/s · p50 **4.86s** | 102.3 req/s · p50 **4.84s** |

차이가 런간 노이즈보다 작다(게이트웨이가 근소하게 빠르게 나온 것도 그 뜻). Caddy는 Go 이벤트 루프 기반이고
SSE는 버퍼링 없이 흘려보낸다. 수 초 단위 생성 앞에서 프록시 한 홉은 사실상 0이다.

### 장애 반경을 줄이는 배치

내부 클라이언트(believe 등)는 **지금처럼 `:11434`로 직접** 호출한다 — 게이트웨이가 죽어도 내부 추론은 멀쩡하다.
게이트웨이는 "외부/도메인 진입"만 담당한다.

```
 외부 ──▶ 도메인:443 ──▶ Caddy ──┬── /v1/*  ──▶ Dynamo :11434   (Bearer 키)
                                 └── 그 외   ──▶ central :8080   (admin 기본인증)
 내부 ───────────────────────────────────────▶ Dynamo :11434   (그대로, 홉 없음)
```

## 인증

| 대상 | 방식 | 값 |
|---|---|---|
| 추론 API (`/v1/*`) | `Authorization: Bearer <키>` | `OMNI_API_KEY` (.env) |
| 대시보드 (그 외 전 경로) | HTTP Basic | `admin` / 같은 키 (해시는 `.env.gateway`) |
| `/health` | 없음 | 업타임 체크용 |
| 워커 에이전트 `:8085/api/internal/*` | `Authorization: Bearer <키>` | `WORKER_API_KEY` (= `OMNI_API_KEY`) |

워커 에이전트는 **키가 설정된 호스트에서만** 검사한다(미설정 호스트는 종전대로 동작). central은 항상 키를 보낸다.
각 워커 호스트에서 켜려면 그 호스트의 레포에서:

```bash
grep -q '^OMNI_API_KEY=' .env || echo 'OMNI_API_KEY="bislaprom3#"' >> .env
docker compose -f docker-compose.worker.yml up -d worker
```

비밀값은 `.env` / `.env.gateway`에 있고 둘 다 git에 올라가지 않는다. 해시의 `$`는 compose 보간 때문에 `$$`로 적는다.

## 도메인 연결 (질문 3)

게이트웨이 하나만 열면 웹과 API가 같은 도메인에서 동작한다.

1. DNS: `omni.example.ac.kr` A 레코드 → 이 사이트의 **공인 IP**
2. 공유기/방화벽 포워딩: **외부 443 → `143.248.74.105:443`** (인증서 발급을 위해 외부 80 → 80도 함께)
3. `.env`에 도메인을 넣고 자동 HTTPS를 켠다:
   ```
   GATEWAY_SITE=omni.example.ac.kr
   CADDY_AUTO_HTTPS=on
   ```
   그리고 `docker-compose.central.yml`의 gateway는 `network_mode: host`이므로 그대로 80/443을 잡는다.
   `docker compose -f docker-compose.central.yml up -d gateway`
4. 결과
   - `https://omni.example.ac.kr/` → 대시보드 (admin 로그인)
   - `https://omni.example.ac.kr/v1/chat/completions` → 추론 API (Bearer 키)
   - OpenAI SDK: `base_url="https://omni.example.ac.kr/v1"`, `api_key="bislaprom3#"`

**외부로는 443(+80)만 포워딩할 것.** 8080·11434·8085·2379·31xxx·33xxx·63xxx·65xxx는 내부 전용이다
(특히 :8085는 GPU 워크로드를 띄우고 죽일 수 있고, :2379는 디스커버리 저장소다).

### 현재 설정 (2026-09-15 적용)

앞단(사내/공유기) 프록시가 TLS를 종료하므로 **게이트웨이는 평문 80번**만 띄운다: `.env`의 `GATEWAY_SITE=:80`,
`CADDY_AUTO_HTTPS=off`. 따라서 포워딩은 **외부 443 → `143.248.74.105:80`** 한 줄이면 되고, 인증서는 앞단에서 관리한다.
Caddy는 `trusted_proxies static private_ranges`로 앞단이 넘기는 `X-Forwarded-For/Proto`를 신뢰해 실제 클라이언트 IP를 남긴다.

검증(모두 `http://143.248.74.105/`):

| 확인 | 결과 |
|---|---|
| `/v1/chat/completions` 키 없이 | 401 |
| `/v1/chat/completions` Bearer 키 | 200 |
| `/` 인증 없이 | 401 |
| `/` admin 로그인 | 200 |
| `/health` | 200 (무인증) |
| 대시보드 API·정적파일·내장 프록시 | 전부 200 |

자체 인증서로 바꾸고 싶어지면 `GATEWAY_SITE=<도메인>` + `CADDY_AUTO_HTTPS=on`으로만 바꾸면 된다(위 4단계 참고).
