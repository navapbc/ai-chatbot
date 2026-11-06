# Lets Braintrust's LLM-as-judge scorers call Vertex AI without a stored
# Google credential. Braintrust presents a signed OIDC token; GCP exchanges it
# for a short-lived Vertex access token. Same shape as the GitHub Actions pool
# in iam.tf — no google_service_account_key anywhere in this repo.
#
# GLOBAL resources, managed only by the 'dev' environment (is_managing_globals).
#
# Values below come from the Braintrust AI provider screen. They are
# identifiers, not credentials — the whole point of WIF is that no Google
# credential is stored in Braintrust. Pinned as locals rather than variables
# for the same reason iam.tf hardcodes 'navapbc': they define a trust boundary
# and must not vary per environment or be overridable by a stray -var.
locals {
  # Our Braintrust organization. Trust is scoped to this org in both the
  # attribute condition and the IAM binding.
  braintrust_org_id = "6c5cfe55-c301-4353-a51a-f471cae4dd8c"

  braintrust_wif_issuer_uri = "https://identity.braintrust.dev"

  braintrust_wif_attribute_mapping = {
    "google.subject"              = "assertion.sub"
    "attribute.braintrust_env"    = "assertion.braintrust_env"
    "attribute.braintrust_org_id" = "assertion.braintrust_org_id"
  }

  braintrust_wif_attribute_condition = "assertion.braintrust_env == 'production' && assertion.braintrust_org_id == '${local.braintrust_org_id}'"

  # Narrower than the pool as a whole: a regression in the condition above
  # cannot silently widen Vertex access to another Braintrust org.
  braintrust_wif_principal = "attribute.braintrust_org_id/${local.braintrust_org_id}"
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

  # Without the condition, any Braintrust tenant could mint tokens for us.
  attribute_condition = local.braintrust_wif_attribute_condition
  attribute_mapping   = local.braintrust_wif_attribute_mapping

  oidc {
    # No jwks_json: Braintrust serves its public keys at the issuer's OIDC
    # discovery endpoint, and the default audience is what Braintrust expects.
    issuer_uri = local.braintrust_wif_issuer_uri
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
  member  = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.braintrust[0].name}/${local.braintrust_wif_principal}"
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
