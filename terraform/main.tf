locals {
  labels = {
    "environment" = "prod"
    "owner"       = "sip-team"
    "repository"  = "n8n"
  }
  dns_records = [
    var.host_name,
  ]
}
