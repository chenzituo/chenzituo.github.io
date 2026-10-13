"""Check local HTML links/assets from one built Hugo page. Standard library only."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set()
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if attrs.get('id'): self.ids.add(attrs['id'])
        if tag in ('a','link') and attrs.get('href'): self.links.append(attrs['href'])
        if tag in ('img','script','iframe') and attrs.get('src'): self.links.append(attrs['src'])

def parse(path):
    p=Links();p.feed(path.read_text());return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output_root',type=Path);ap.add_argument('page');a=ap.parse_args()
    root=a.output_root.resolve();page=root/a.page;doc=parse(page);errors=[];checked=0
    for link in doc.links:
        u=urlsplit(link)
        if u.scheme not in ('','http','https'):continue
        if u.netloc and u.hostname not in ('localhost','127.0.0.1','chenzituo.github.io'):continue
        raw=unquote(u.path)
        target=(root/raw.lstrip('/')) if raw.startswith('/') else (page.parent/raw if raw else page)
        if target.is_dir():target=target/'index.html'
        if not target.exists():errors.append(f'Missing: {link}');continue
        checked+=1
        if u.fragment and target.suffix=='.html' and unquote(u.fragment) not in parse(target).ids:
            errors.append(f'Missing fragment: {link}')
    print(f'{checked} local targets checked; {len(errors)} errors')
    for e in errors:print(e)
    raise SystemExit(bool(errors))
if __name__=='__main__':main()
