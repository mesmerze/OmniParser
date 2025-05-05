locals {
  labels = {
    "environment" = "prod"
    "owner"       = "sip-team"
    "repository"  = "omniparser"
  }
  dns_records = [
    var.host_name,
  ]
}
