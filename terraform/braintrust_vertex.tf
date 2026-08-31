# Lets Braintrust's LLM-as-judge scorers call Vertex AI without a stored
# Google credential. Braintrust presents a signed OIDC token; GCP exchanges it
# for a short-lived Vertex access token. Same shape as the GitHub Actions pool
# in iam.tf — no google_service_account_key anywhere in this repo.
#
# GLOBAL resources, managed only by the 'dev' environment (is_managing_globals).
#
# Two-pass setup: Braintrust shows the issuer, attribute mapping, and condition
# only after you start creating the provider in its UI. Fill the variables in
# terraform.tfvars from that screen, apply, then finish the Braintrust side with
# the pool/provider IDs below.

# Enabling without the Braintrust-supplied values would create a pool that
# trusts nothing (or worse, everything). Fail the plan instead.
resource "terraform_data" "braintrust_wif_preconditions" {
  count = var.braintrust_wif_enabled ? 1 : 0

  lifecycle {
    precondition {
      condition = alltrue([
        var.braintrust_wif_issuer_uri != "",
        length(var.braintrust_wif_attribute_mapping) > 0,
        var.braintrust_wif_attribute_condition != "",
        var.braintrust_wif_principal_attribute != "",
      ])
      error_message = "braintrust_wif_enabled requires issuer_uri, attribute_mapping, attribute_condition, and principal_attribute (all from the Braintrust AI provider screen)."
    }
  }
}

resource "google_iam_workload_identity_pool" "braintrust" {
  count = local.is_managing_globals && var.braintrust_wif_enabled ? 1 : 0

  workload_identity_pool_id = "braintrust-pool"
  display_name              = "Braintrust Pool"
  description               = "Workload Identity Pool for Braintrust LLM-as-judge scorers"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_iam_workload_identity_pool_provider" "braintrust" {
  count = local.is_managing_globals && var.braintrust_wif_enabled ? 1 : 0

  workload_identity_pool_id          = google_iam_workload_identity_pool.braintrust[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "braintrust-provider"
  display_name                       = "Braintrust Provider"
  description                        = "OIDC provider for Braintrust-signed tokens"

  # Both come from the Braintrust AI provider setup screen. The condition scopes
  # the trust to our org — without it any Braintrust tenant could mint tokens.
  attribute_condition = var.braintrust_wif_attribute_condition
  attribute_mapping   = var.braintrust_wif_attribute_mapping

  oidc {
    # No jwks_json: Braintrust serves its public keys at the issuer's OIDC
    # discovery endpoint, and the default audience is what Braintrust expects.
    issuer_uri = var.braintrust_wif_issuer_uri
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Vertex access for the federated identity. aiplatform.user is what Braintrust's
# docs require; it does not grant access to any other Google API.
resource "google_project_iam_member" "braintrust_vertex_user" {
  count = local.is_managing_globals && var.braintrust_wif_enabled ? 1 : 0

  project = local.project_id
  role    = "roles/aiplatform.user"
  member  = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.braintrust[0].name}/${var.braintrust_wif_principal_attribute}"
}

# Paste these into the Braintrust provider screen to finish setup.
output "braintrust_wif_pool_id" {
  description = "Workload identity pool ID for the Braintrust AI provider config."
  value       = try(google_iam_workload_identity_pool.braintrust[0].workload_identity_pool_id, null)
}

output "braintrust_wif_provider_id" {
  description = "Workload identity provider ID for the Braintrust AI provider config."
  value       = try(google_iam_workload_identity_pool_provider.braintrust[0].workload_identity_pool_provider_id, null)
}
