locals {
  labels = {
    "environment" = "prod"
    "owner"       = "sip-team"
    "repository"  = "omniparser"
  }
  dns_records = {
    (var.host_name_cloud) = data.google_compute_address.ingress.address
    (var.host_name_hetzner) = "138.201.11.225"                              
  }
}
