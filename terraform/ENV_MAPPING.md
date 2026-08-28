# Environment Variable Mapping

This document shows how environment variables from your local `.env` files map to the Terraform-managed Cloud Run deployment.

## ✅ **Secret Manager Mapping**

| Local Env Var | Secret Manager Name | Service |
|---------------|-------------------|---------|
| `OPENAI_API_KEY` | `openai-api-key` | Both |
| `ANTHROPIC_API_KEY` | `anthropic-api-key` | Both |
| `EXA_API_KEY` | `exa-api-key` | Both |
| `GOOGLE_GENERATIVE_AI_API_KEY` | `google-generative-ai-key` | Both |
| `GROK_API_KEY` | `grok-api-key` | Mastra |
| `XAI_API_KEY` | `xai-api-key` | Chatbot |
| `DATABASE_URL` | `database-url-{env}` | Both |
| `POSTGRES_URL` | `postgres-url` | Chatbot |
| `MASTRA_JWT_SECRET` | `mastra-jwt-secret` | Mastra |
| `MASTRA_APP_PASSWORD` | `mastra-app-password` | Mastra |
| `MASTRA_JWT_TOKEN` | `mastra-jwt-token` | Both |
| `AUTH_SECRET` | `auth-secret` | Chatbot |
| `GOOGLE_APPLICATION_CREDENTIALS` | `vertex-ai-credentials` | Both |
| `APRICOT_API_BASE_URL` | `apricot-api-base-url` | Both |
| `APRICOT_CLIENT_ID` | `apricot-client-id` | Both |
| `APRICOT_CLIENT_SECRET` | `apricot-client-secret` | Both |

## 🔧 **Computed Environment Variables**

| Variable | Mastra Service | Chatbot Service |
|----------|---------------|-----------------|
| `PLAYWRIGHT_MCP_URL` | `http://{vm-internal-ip}:8931/mcp` | `http://{vm-internal-ip}:8931/mcp` |
| `BROWSER_STREAMING_URL` | `ws://{vm-internal-ip}:8933` | `ws://{vm-internal-ip}:8933` |
| `BROWSER_STREAMING_HOST` | - | `{vm-internal-ip}` |
| `BROWSER_STREAMING_PORT` | - | `8933` |
| `NEXT_PUBLIC_MASTRA_SERVER_URL` | - | `{mastra-cloud-run-url}` |
| `GCS_BUCKET_NAME` | - | `labs-asp-artifacts-{env}` |
| `NEXTAUTH_URL` | - | `https://{domain}` |

## 📋 **Static Configuration**

| Variable | Value | Service |
|----------|-------|---------|
| `NODE_ENV` | `production/development` | Both |
| `ENVIRONMENT` | `dev/preview/prod` | Both |
| `GCP_PROJECT_ID` | `nava-labs` | Both |
| `GOOGLE_CLOUD_PROJECT` | `nava-labs` | Both |
| `GOOGLE_VERTEX_PROJECT` | `nava-labs` | Both |
| `GOOGLE_VERTEX_LOCATION` | `global` | Both |
| `CORS_ORIGINS` | `https://{domain}` or `*` | Mastra |

## 🏗️ **Architecture Flow**

```
┌─────────────────────────────────────────────────────────┐
│                Browser VM                               │
│  - Playwright MCP Server (8931)                        │
│  - Browser Streaming WebSocket (8933)                  │
│  - Internal IP: {computed at runtime}                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Mastra Cloud Run                         │
│  - Connects to Browser VM via internal IP              │
│  - Serves /chat endpoint for web-automation-model      │
│  - All AI API keys + Mastra authentication             │
│  - URL: https://mastra-app-{env}-{project}.run.app     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               AI Chatbot Cloud Run                      │
│  - Next.js frontend with conditional routing           │
│  - Connects to Mastra service for web-automation       │
│  - Connects to Browser VM for direct streaming         │
│  - All AI API keys + auth + dual database access       │
│  - URL: https://ai-chatbot-{env}-{project}.run.app     │
└─────────────────────────────────────────────────────────┘
```

## 🔐 **Security Notes**

1. **All sensitive keys stored in Secret Manager** - no hardcoded secrets
2. **Service accounts** with least-privilege access
3. **Internal networking** between services via GCP internal IPs
4. **Firewall rules** restrict browser VM access to internal ranges only
5. **CORS configuration** environment-specific (prod = domain only, dev = wildcard)

## 🚀 **Deployment Readiness**

All environment variables from your existing `.env` files are now:
- ✅ **Mapped to Secret Manager** (18 secrets already created)
- ✅ **Configured in Terraform** for both services
- ✅ **Computed dynamically** for networking between services
- ✅ **Environment-specific** (dev/preview/prod support)

The infrastructure is ready for deployment with your existing Secret Manager setup!