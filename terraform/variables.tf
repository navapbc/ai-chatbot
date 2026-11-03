# Project configuration
variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "nava-labs"
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for VM instances"
  type        = string
  default     = "us-central1-a"
}

# Domain configuration
variable "domain_name" {
  description = "Base domain name for the application"
  type        = string
  default     = "labs-asp.navateam.com"
}

variable "enable_custom_domain" {
  description = "Enable custom domain mapping (requires domain verification via Google Search Console)"
  type        = bool
  default     = false # Default to false to avoid verification issues
}

# AI Chatbot Cloud Run configuration
variable "chatbot_image_url" {
  description = "Container image URL for AI chatbot service"
  type        = string
  default     = "us-central1-docker.pkg.dev/nava-labs/labs-asp/ai-chatbot:latest"
}

variable "chatbot_cpu" {
  description = "CPU allocation for AI Chatbot Cloud Run service"
  type        = string
  default     = "2" # 2 vCPUs for better concurrent request handling
}

variable "chatbot_memory" {
  description = "Memory allocation for AI Chatbot Cloud Run service"
  type        = string
  default     = "8Gi"
}

variable "chatbot_min_instances" {
  description = "Minimum instances for AI Chatbot service"
  type        = number
  default     = 2 # Keep instances warm to avoid cold starts during load testing
}

variable "chatbot_max_instances" {
  description = "Maximum instances for AI Chatbot service"
  type        = number
  default     = 20
}

variable "chatbot_timeout" {
  description = "Request timeout for AI Chatbot service in seconds"
  type        = number
  default     = 3600 # 1 hour maximum for AI agent workflows with browser automation
}

# Environment selection
variable "environment" {
  description = "Environment to deploy (dev, preview, preview-pr-N, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = can(regex("^(dev|preview|preview-pr-[0-9]+|prod)$", var.environment))
    error_message = "Environment must be one of: dev, preview, preview-pr-NUMBER, or prod"
  }
}

# VPC Network Configuration
# These CIDR blocks should be defined per workspace in GitHub Actions
variable "vpc_cidr_public" {
  description = "CIDR block for public subnet (pattern: 10.X.0.0/16)"
  type        = string
  default     = "10.0.0.0/20" # Default for dev if not overridden
}

variable "vpc_cidr_private" {
  description = "CIDR block for private subnet (pattern: 10.X.16.0/16)"
  type        = string
  default     = "10.0.16.0/20" # Default for dev if not overridden
}

variable "vpc_cidr_db" {
  description = "CIDR block for database subnet (pattern: 10.X.32.0/16)"
  type        = string
  default     = "10.0.32.0/20" # Default for dev if not overridden
}

variable "vpc_connector_cidr" {
  description = "CIDR block for VPC Serverless Connector (must be /28)"
  type        = string
  default     = "10.0.48.0/28" # Default for dev if not overridden
  validation {
    condition     = can(cidrhost(var.vpc_connector_cidr, 0)) && tonumber(split("/", var.vpc_connector_cidr)[1]) == 28
    error_message = "VPC connector CIDR must be a /28 block"
  }
}

# Firewall Configuration
# Granular firewall rules per service with configurable ports
# Database passwords are stored in Secret Manager:
# - nava-db-password-dev (for dev environment)
# - database-password-prod (for prod environment)
# No variable needed - passwords are retrieved from Secret Manager

# Note: Preview environments use shared VPC with permanent peering to dev VPC
# No need for dynamic VPC tracking variables

# Feature flag for guest login (bypasses OAuth in preview environments)
variable "use_guest_login" {
  description = "Feature flag to enable guest login form for preview environments"
  type        = string
  default     = "false"
}

# Braintrust -> Vertex AI workload identity federation. The pool config itself
# lives in braintrust_vertex.tf; this only gates whether it is created.
variable "braintrust_wif_enabled" {
  description = "Create the Braintrust workload identity pool and grant it Vertex access"
  type        = bool
  default     = true
}

variable "jev_features" {
  description = "Comma-separated Jev features to enable in the agent loop (gap-triage, summary-check, field-value), or \"all\". Empty disables every Jev call. See lib/jev/client.ts."
  type        = string
  # Off unless a caller opts in. Callers that pass no value get this default:
  # promote.yml is one (it passes only environment and chatbot_image_url), so
  # a non-empty default would silently enable Jev on every promotion. Preview
  # opts in through the explicit -var in deploy.yml.
  default = ""
}

# Browser relay (relay.tf)
variable "relay_image_url" {
  description = "Container image URL for the browser relay service. Empty skips the relay service and the app's relay env vars."
  type        = string
  # Empty so callers that pass nothing still plan. A caller that passes nothing
  # against a state that already holds the relay would plan to destroy it, so
  # both deploy.yml and promote.yml pass the image built for the same commit
  # whenever the repository variable RELAY_ENABLED is 'true' (and "" otherwise,
  # which keeps every relay resource and secret reference out of the plan).
  default = ""
}

variable "relay_max_instances" {
  description = "Maximum instances for the browser relay service"
  type        = number
  default     = 1
  # Runs live in one process's memory. More than one instance needs a shared
  # REDIS_URL plus RELAY_MULTI_INSTANCE_ACKNOWLEDGED=1 (see relay/env.ts for the
  # known cross-instance gaps), and relay.tf sets neither.
  validation {
    condition     = var.relay_max_instances == 1
    error_message = "relay_max_instances must be 1: more instances need REDIS_URL and RELAY_MULTI_INSTANCE_ACKNOWLEDGED=1, which relay.tf does not configure."
  }
}

variable "relay_exclude_request_logs" {
  description = "Exclude the relay service's Cloud Run request logs, whose URLs carry run tokens (see relay.tf)"
  type        = bool
  default     = true
}

# Pilot settings (plan 5). Every default below is today's behavior: nothing
# set, relay runs off. deploy.yml and promote.yml both set them through
# scripts/relay-terraform-vars.sh, which turns the GitHub environment-level
# variables RELAY_RUNS_ENABLED and the rest into TF_VAR_ entries (TF_VAR_, since
# older tags do not declare them). A new pilot variable needs that script, not
# a -var= in one workflow, or the other workflow plans it back to its default.

variable "relay_min_instances" {
  description = "Minimum instances for the browser relay service. 0 (default) scales to zero; the pilot sets 1, since a cold start or recycle drops every live run's sockets and in-memory run state."
  type        = number
  default     = 0
  validation {
    condition     = contains([0, 1], var.relay_min_instances)
    error_message = "relay_min_instances must be 0 or 1 (the relay runs as a single instance)."
  }
}

variable "relay_runs_enabled" {
  description = "Set RELAY_RUNS_ENABLED=1 on the app (extension relay runs). Takes effect only where the relay is deployed (relay_image_url set)."
  type        = bool
  default     = false
}

variable "relay_token_endpoint_enabled" {
  description = "Set RELAY_TOKEN_ENDPOINT_ENABLED=1 on the app, enabling /api/relay/token for the side panel. Takes effect only where the relay is deployed."
  type        = bool
  default     = false
}

variable "relay_extension_origin" {
  description = "RELAY_EXTENSION_ORIGIN on the app, the full chrome-extension://<id> origin the relay routes accept. Empty (default) leaves it unset, which means no Origin check. Takes effect only where the relay is deployed, and must match extension/EXTENSION_ID (precondition in relay.tf)."
  type        = string
  default     = ""
  validation {
    condition     = var.relay_extension_origin == "" || can(regex("^chrome-extension://[a-p]{32}$", var.relay_extension_origin))
    error_message = "relay_extension_origin must be empty or chrome-extension://<32 characters a-p>."
  }
}

variable "relay_allowed_models" {
  description = "RELAY_ALLOWED_MODELS on the app (comma-separated picker ids a relay run may request). Empty (default) leaves it unset, so the app's built-in default applies. Takes effect only where the relay is deployed."
  type        = string
  default     = ""
}

variable "relay_allowed_hosts" {
  description = <<-EOT
    RELAY_ALLOWED_HOSTS on the relay: comma-separated exact Host values (bare
    host names, no scheme or path; the service uri output starts with https://,
    so strip it). Empty (default) leaves it unset. A wrong host makes EVERY
    relay request answer 421, so switch in two applies, while relay runs are
    still off: first set this while relay_allowed_host_suffixes still holds
    ".run.app"; then, in a second apply, set relay_allowed_host_suffixes = ""
    to drop the suffix (in the workflows: RELAY_ALLOWED_HOST_SUFFIXES=none). A request check made while the suffix is set proves
    nothing (a host passes on the list OR the suffix). Before the second
    apply, confirm the relay's RELAY_ALLOWED_HOSTS contains exactly the host
    in the app's RELAY_PUBLIC_URL (the relay service uri, output
    relay_service_url). See docs/rollout-pilot.md.
  EOT
  type        = string
  default     = ""
  validation {
    condition = var.relay_allowed_hosts == "" || alltrue([
      for host in split(",", var.relay_allowed_hosts) :
      trimspace(host) != "" && can(regex("^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$", lower(trimspace(host))))
    ])
    error_message = "relay_allowed_hosts must be empty or comma-separated bare host names (letters, digits, dots and hyphens) with no empty entry, scheme, port, path or space."
  }
}

variable "relay_allowed_host_suffixes" {
  description = "RELAY_ALLOWED_HOST_SUFFIXES on the relay. Default \".run.app\" is today's value; set \"\" (second apply of the exact-host switch, see relay_allowed_hosts) to leave it unset."
  type        = string
  default     = ".run.app"
  validation {
    condition = var.relay_allowed_host_suffixes == "" || alltrue([
      for suffix in split(",", var.relay_allowed_host_suffixes) : trimspace(suffix) != ""
    ])
    error_message = "relay_allowed_host_suffixes must not contain an empty entry."
  }
}

variable "relay_command_duration_metric_enabled" {
  description = "Create the log-based distribution metric on the relay's command durationMs (relay.tf). Needs logging.logMetrics.create on the deploy identity (roles/logging.configWriter)."
  type        = bool
  default     = false
}
