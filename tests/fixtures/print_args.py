from __future__ import annotations

import json
import sys

print(json.dumps(sys.argv[1:], ensure_ascii=False))
