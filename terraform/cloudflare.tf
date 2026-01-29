# Create a DNS record in Cloudflare for Omniparser endpoints
resource "cloudflare_record" "record" {
  for_each = local.dns_records
  zone_id  = data.cloudflare_zone.name.id
  name     = each.key
  content  = each.value
  type     = "A"
  proxied  = false
}
