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