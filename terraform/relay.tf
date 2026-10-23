# Browser relay - bridges agent-browser (in the app container) to the bridge
# extension in a caseworker's browser. Image: relay/Dockerfile.
#
# Skipped entirely when relay_image_url is empty, and the app container's relay
# env vars in cloud_run.tf follow the same switch.

locals {
  relay_enabled = var.relay_image_url != ""

  # Suffixed with base_environment, not environment: every preview-pr-N shares
  # the "-preview" secrets, the same way database-url-preview is shared.
  # Per-PR secret names could not be created ahead of a PR's first deploy, and a
  # missing secret fails the revision. The owner creates these (Terraform never
  # holds the values): relay-token-secret-{dev,preview,prod} (>= 32 chars) and
  # eve-internal-secret-{dev,preview,prod}.
  relay_token_secret_name  = "relay-token-secret-${local.base_environment}"
  eve_internal_secret_name = "eve-internal-secret-${local.base_environment}"
}

resource "google_cloud_run_v2_service" "relay" {
  count = local.relay_enabled ? 1 : 0

  name     = local.env_config.relay_service_name
  location = local.region

  # Disable deletion protection for preview environments to allow easy cleanup
  deletion_protection = startswith(var.environment, "preview-") ? false : true

  template {
    # Reads relay-token-secret-* through the project-wide
    # roles/secretmanager.secretAccessor grant in cloud_run.tf.
    service_account = google_service_account.cloud_run.email

    # No vpc_access: the relay talks only to its two WebSocket peers and needs
    # neither the database nor the NAT's static IP.

    containers {
      image = var.relay_image_url

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      # PORT is injected by Cloud Run; relay/main.ts reads it.

      env {
        name = "RELAY_TOKEN_SECRET"
        value_source {
          secret_key_ref {
            secret  = local.relay_token_secret_name
            version = "latest"
          }
        }
      }

      env {
        name  = "RELAY_ALLOWED_HOST_SUFFIXES"
        value = ".run.app"
      }

      env {
        name  = "RELAY_EXTENSION_ID"
        value = trimspace(file("${path.module}/../extension/EXTENSION_ID"))
      }

      # Deliberately no REDIS_URL / RELAY_MULTI_INSTANCE_ACKNOWLEDGED: the
      # supported deployment is one instance with the in-process bus (see
      # relay_max_instances in variables.tf).

      ports {
        container_port = 8080
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = var.relay_max_instances
    }

    # Each run is two long-lived WebSockets (extension and agent) that must
    # meet in one process; with a single instance there is nothing to pin.
    session_affinity = false

    # Cloud Run's maximum. A WebSocket is a request, so this caps how long one
    # run's connection lasts before the client must reconnect.
    timeout = "3600s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = merge(local.common_labels, {
    environment = var.environment
    component   = "browser-relay"
  })

  depends_on = [
    google_project_service.required_apis,
    google_service_account.cloud_run,
  ]
}

# Public invoker: the extension connects from a caseworker's browser and
# cannot attach a Google identity token, so Cloud Run IAM cannot be the gate.
# Every WebSocket must instead present a run token signed with
# RELAY_TOKEN_SECRET; the relay refuses the upgrade otherwise.
resource "google_cloud_run_v2_service_iam_member" "relay_public_access" {
  count = local.relay_enabled ? 1 : 0

  name     = google_cloud_run_v2_service.relay[0].name
  location = google_cloud_run_v2_service.relay[0].location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
