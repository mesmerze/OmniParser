output "domain_name" {
  value       = var.domain_name
  sensitive   = false
  description = "Domain name for the service"
}

output "host_name_cloud" {
  value       = var.host_name_cloud
  sensitive   = false
  description = "Host name for the service"
}

output "host_name_hetzner" {
  value       = var.host_name_hetzner
  sensitive   = false
  description = "Host name for the service"
}
