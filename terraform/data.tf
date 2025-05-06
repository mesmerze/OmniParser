data "google_client_config" "default" {}
data "google_container_cluster" "gke" { name = var.gke_cluster_name }

# Data source to get the IP address of the ingress
data "google_compute_address" "ingress" { name = var.ingress }

# Cloudflare data source to get token and zone ID
data "google_secret_manager_secret_version" "cloudflare_token" { secret = "cloudflare_token" }
data "cloudflare_zone" "name" { name = var.domain_name }

