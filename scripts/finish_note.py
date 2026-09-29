#!/usr/bin/env python3
"""Mark a written note complete and stamp its publication date (stdlib only)."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '**Coming soon.** This is a topic I plan to return to and write about.'


def finish(text, summary, timestamp):
    parts = text.split('---', 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError('Expected YAML front matter delimited by --- at the start.')
    header, body = parts[1], parts[2]
    if re.search(r'^completed:\s*true\s*$', header, re.M | re.I):
        raise ValueError('Already completed; keeping its original publication date.')
    if not body.strip() or PLACEHOLDER in body:
        raise ValueError('Replace the Coming soon placeholder with your finished note first.')
    if not summary.strip():
        raise ValueError('Provide a nonempty summary.')
    # Preserve author-written metadata; only change the completion-related fields.
    tag_match = re.search(r'^tags:\s*\[([^\]\n]*)\][ \t]*$', header, re.M)
    if re.search(r'^tags:', header, re.M) and not tag_match:
        raise ValueError('Use inline tags, e.g. tags: [planned, physics], before finishing.')
    if tag_match:
        tags = [x.strip() for x in tag_match[1].split(',') if x.strip()]
        tags = [x for x in tags if x.strip('\"\'').lower() != 'planned']
        header = header[:tag_match.start()] + 'tags: [' + ', '.join(tags) + ']' + header[tag_match.end():]
    updates = {
        'date': timestamp, 'publishDate': timestamp, 'completed': 'true',
        'draft': 'false', 'weight': '-1', 'summary': json.dumps(summary.strip()),
        'showtoc': 'true', 'ShowReadingTime': 'true',
    }
    for key, value in updates.items():
        pattern = r'^' + re.escape(key) + r':[^\n]*$'
        line = f'{key}: {value}'
        if re.search(pattern, header, re.M | re.I):
            header = re.sub(pattern, lambda _: line, header, flags=re.M | re.I)
        else:
            header = header.rstrip() + '\n' + line + '\n'
    return '---' + header + '---' + body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug', help='Folder name under content/posts')
    parser.add_argument('--summary', required=True, help='Summary for the Posts page')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.slug):
        parser.error('Use a note folder name, such as physical-trajectory-modeling.')
    path = ROOT / 'content' / 'posts' / args.slug / 'index.md'
    timestamp = datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
    try:
        updated = finish(path.read_text(), args.summary, timestamp)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    path.write_text(updated)
    print(f'Completed {args.slug} at {timestamp}. Preview, commit, and push to publish.')


if __name__ == '__main__':
    main()
