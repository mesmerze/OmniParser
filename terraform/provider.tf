terraform {
  required_version = ">= 1.8.3"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.25.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.36.0"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.52.0"
    }
  }
  backend "gcs" {
    bucket = "tf-omniparser-jepfdfdc859xft11"
    prefix = "omniparser"
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.gke.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.google_container_cluster.gke.master_auth[0].cluster_ca_certificate)
}

provider "cloudflare" {
  api_token = data.google_secret_manager_secret_version.cloudflare_token.secret_data
}
