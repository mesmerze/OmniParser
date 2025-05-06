variable "gcp_region" {
  description = "GCP Region"
  default     = "europe-west3"
}

variable "gcp_region_zone" {
  description = "GCP Zone"
  default     = "europe-west3-a"
}

variable "gcp_project" {
  default = "oi-tset"
}

variable "gke_cluster_name" {
  type        = string
  default     = "private-cluster"
  description = "GKE cluster name"
}

variable "domain_name" {
  type        = string
  default     = "test.io"
  description = "Domain name for the service"
}

variable "host_name" {
  type        = string
  default     = "omniparser"
  description = "Host name for the service"
}

variable "ingress" {
  type        = string
  default     = "istio-external-static-ip"
  description = "Ingress IP address name to which DNS records are point"
}
