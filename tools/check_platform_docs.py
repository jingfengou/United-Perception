"""Check tracked Markdown/HTML guides, source links and HTML anchor targets."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_BLOB = 'https://github.com/jingfengou/United-Perception/blob/codex/up-platform-clean/'


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.duplicate_ids = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.links.append(value)
            if name == 'id' and value:
                if value in self.ids:
                    self.duplicate_ids.add(value)
                self.ids.add(value)


def main():
    parser = MarkdownIt()
    failures, count = [], 0
    tracked = {ROOT/name for name in subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT).decode('utf8').split('\0') if name}
    published = tracked | {parent for path in tracked for parent in path.parents if parent.is_relative_to(ROOT)}
    files = sorted(path for path in tracked if path == ROOT/'README.md' or (
        path.is_relative_to(ROOT/'docs/platform') and path.suffix in ('.md', '.html')))
    for path in files:
        links = []
        html_document = None
        if path.suffix == '.html':
            html_document = HtmlLinks()
            html_document.feed(path.read_text(encoding='utf8'))
            links = html_document.links
            for identifier in sorted(html_document.duplicate_ids):
                failures.append(f'{path.relative_to(ROOT)}: duplicate HTML id {identifier}')
        else:
            for token in parser.parse(path.read_text(encoding='utf8')):
                for child in [token, *(token.children or [])]:
                    for key in ('href', 'src'):
                        if child.attrGet(key):
                            links.append(child.attrGet(key))
                    if child.type in ('html_block', 'html_inline'):
                        html = HtmlLinks()
                        html.feed(child.content)
                        links.extend(html.links)
        for link in links:
            url = urlsplit(link)
            if html_document is not None and link.startswith(REPOSITORY_BLOB):
                target = (ROOT/unquote(link[len(REPOSITORY_BLOB):].split('#', 1)[0])).resolve()
            elif html_document is not None and not url.path and url.fragment:
                count += 1
                if unquote(url.fragment) not in html_document.ids:
                    failures.append(f'{path.relative_to(ROOT)}: missing HTML anchor {link}')
                continue
            elif url.scheme or url.netloc or not url.path:
                continue
            else:
                target = (path.parent/unquote(url.path)).resolve()
            count += 1
            if not target.is_relative_to(ROOT) or not target.exists() or target not in published:
                failures.append(f'{path.relative_to(ROOT)}: missing, untracked or external local path {link}')
    for failure in failures:
        print(failure)
    print(f'{len(files)} documents, {count} local links, {len(failures)} failures')
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
