import urllib.request, urllib.parse, re

query = "Osmania University B.Tech Academic Regulations pdf"
url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=15) as r:
    html = r.read().decode("utf-8", errors="ignore")

# Show a result block raw
idx = html.find('result__a')
print("=== RAW BLOCK AROUND result__a ===")
print(html[max(0,idx-50):idx+400])
print()

# Try the correct pattern - DDG uses uddg= for redirects
# Pattern 1: href with uddg param
urls_uddg = re.findall(r'uddg=([^&"\'>\s]+)', html)
print(f"uddg URLs found: {len(urls_uddg)}")
for u in urls_uddg[:5]:
    print(" -", urllib.parse.unquote(u)[:100])

print()

# Pattern 2: titles  
titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.S)
print(f"Titles found: {len(titles)}")
for t in titles[:5]:
    clean = re.sub(r'<[^>]+>', '', t).strip()
    print(" -", clean[:80])

print()

# Pattern 3: snippets
snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)>', html, re.S)
print(f"Snippets found: {len(snippets)}")
for s in snippets[:3]:
    clean = re.sub(r'<[^>]+>', '', s).strip()
    print(" -", clean[:100])
