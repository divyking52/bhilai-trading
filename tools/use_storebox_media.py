"""Points every image in index.html at the Storebox CDN.

Reads the asset map (local relative path -> Storebox file_url) produced from the Storebox media API.
"""
import json
import os
import re
import sys

CDN = 'https://cdn.storebox.ai'
html_path = 'index.html'
mapping = json.load(open(sys.argv[1]))

html = open(html_path, encoding='utf-8').read()

# Absolute social/SEO image URLs.
html = html.replace('https://www.bhilaitradingco.com/assets/media/og-logo.jpg', CDN + mapping['og-logo.jpg'])

# Static attributes: assets/media/<path>.
def static(m):
    rel = m.group(1)
    return CDN + mapping[rel] if rel in mapping else m.group(0)

html = re.sub(r'assets/media/([A-Za-z0-9_\-/]+\.(?:webp|png|jpe?g))', static, html)

# Runtime lookup for data-driven images.
js_map = json.dumps(dict(sorted(mapping.items())), indent=2).replace('\n', '\n  ').replace('"', "'")
html = html.replace(
    "const MEDIA = 'assets/media/';",
    "const MEDIA_CDN = '%s';\n"
    "/* Storebox media library: local asset path -> hosted file. */\n"
    "const MEDIA_FILES = %s;\n"
    "const media = p => (MEDIA_FILES[p] ? MEDIA_CDN + MEDIA_FILES[p] : 'assets/media/' + p);" % (CDN, js_map)
)
html = re.sub(r'\$\{MEDIA\}([^"`\'\s<>]*?\.(?:webp|png|jpe?g))', lambda m: '${media(`' + m.group(1) + '`)}', html)

# Connect early to the CDN.
html = html.replace(
    '<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>',
    '<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>\n<link rel="preconnect" href="https://cdn.storebox.ai">'
)

open(html_path, 'w', encoding='utf-8', newline='\n').write(html)
left = re.findall(r'assets/media/[^\'"`\s)]*', html)
print('remaining local refs:', sorted(set(left)))
print('MEDIA template refs left:', len(re.findall(r'\$\{MEDIA\}', html)))
print('cdn refs:', html.count(CDN))
