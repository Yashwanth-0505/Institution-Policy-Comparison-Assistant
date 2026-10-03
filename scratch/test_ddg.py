import urllib.request
import urllib.parse
import re

query = "Neil Gogte Institute of Technology Academic Regulations pdf"
url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
headers = {
    "User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)"
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode("utf-8", errors="ignore")

results = []
for m in re.finditer(r'<a[^>]+href="/url\?q=([^&"]+)[^"]*"[^>]*>(.*?)</a>', html, re.S):
    href, raw_text = m.group(1), m.group(2)
    clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()
    clean_url = urllib.parse.unquote(href)
    if not "google.com" in clean_url and clean_text and clean_url.startswith("http"):
        results.append({"title": clean_text, "url": clean_url})

print(f"Extracted {len(results)} results using classic User-Agent:")
for r in results[:5]:
    print(" -> Title:", r['title'])
    print("    URL:", r['url'])
