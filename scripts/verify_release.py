"""Check every distributed file against the public integrity manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release_manifest.json').read_text())
for name,row in manifest['files'].items():
    data=(root/name).read_bytes()
    assert len(data)==row['bytes'],name
    assert hashlib.sha256(data).hexdigest()==row['sha256'],name
print('Verified',len(manifest['files']),'public files. Integrity is not external scientific certification.')
