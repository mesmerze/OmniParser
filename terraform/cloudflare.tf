# Create a DNS record in Cloudflare for Omniparser endpoints
resource "cloudflare_record" "record" {
  for_each = toset(local.dns_records)
  zone_id  = data.cloudflare_zone.name.id
  name     = each.key
  content  = data.google_compute_address.ingress.address
  type     = "A"
  proxied  = false
}
