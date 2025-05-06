output "domain_name" {
  value       = var.domain_name
  sensitive   = false
  description = "Domain name for the service"
}

output "host_name" {
  value       = var.host_name
  sensitive   = false
  description = "Host name for the service"
}
