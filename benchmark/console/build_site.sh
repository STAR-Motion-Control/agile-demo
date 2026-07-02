#!/usr/bin/env bash
# Assemble the experiment-console site on the 4090.
#
# Expects console files synced to $CONSOLE_SRC (scp/rsync from the mac):
#   index.html  manifest.json  server.py  [PROTOCOL.md]
# and benchmark videos already rsynced into $SITE/videos/<model>/.
#
# Usage (on the 4090):
#   bash /sda/lizhe/g1bench/console/build_site.sh
set -euo pipefail

CONSOLE_SRC="${CONSOLE_SRC:-/sda/lizhe/g1bench/console}"
SITE="${SITE:-/sda/lizhe/g1bench/site}"

mkdir -p "$SITE/videos/custom" "$SITE/jobs"

for f in index.html manifest.json; do
  if [ ! -f "$CONSOLE_SRC/$f" ]; then
    echo "ERROR: missing $CONSOLE_SRC/$f (sync console files first)" >&2
    exit 1
  fi
  cp -f "$CONSOLE_SRC/$f" "$SITE/$f"
done
[ -f "$CONSOLE_SRC/PROTOCOL.md" ] && cp -f "$CONSOLE_SRC/PROTOCOL.md" "$SITE/PROTOCOL.md"

# Normalize manifest video paths to the site layout (videos/<model>/<file>)
# in case a non --site manifest was synced (idempotent for --site manifests).
python3 - "$SITE/manifest.json" <<'PY'
import json, re, sys
path = sys.argv[1]
with open(path) as f:
    m = json.load(f)
fixed = []
for v in m.get("videos", []):
    rel = re.sub(r"^(\.\./)*results/videos_r2plus/", "videos/", v["rel_path"])
    fixed.append(dict(v, rel_path=rel))
out = dict(m, videos=fixed,
           protocol_doc=re.sub(r"^(\.\./)+", "", m.get("protocol_doc") or "PROTOCOL.md"))
with open(path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("manifest normalized: %d videos" % len(fixed))
PY

# Sanity: every manifest video must exist under the site root.
python3 - "$SITE" <<'PY'
import json, os, sys
site = sys.argv[1]
with open(os.path.join(site, "manifest.json")) as f:
    m = json.load(f)
missing = [v["rel_path"] for v in m["videos"]
           if not os.path.exists(os.path.join(site, v["rel_path"]))]
print("site video check: %d/%d present, %d missing"
      % (len(m["videos"]) - len(missing), len(m["videos"]), len(missing)))
for p in missing[:10]:
    print("  missing:", p)
sys.exit(1 if missing else 0)
PY

echo "site assembled at $SITE"
