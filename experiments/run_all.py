"""Run the fixed GPU experiment and then its closed-set term router.
Requires the original prepared data and core; see docs/reproduction.md.
"""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
main=(root/'experiments/run_extension.py').read_text()
router=(root/'experiments/run_router.py').read_text()
scope={'__name__':'__main__', 'EXTENDED_RUNNER_SOURCE':main,
       'EXTENDED_PROTOCOL_SOURCE':(root/'docs/protocole_extension_mce.md').read_text(),
       'ROUTER_RUNNER_SOURCE':router}
exec(compile(main,'run_extension.py','exec'),scope)
exec(compile(router,'run_router.py','exec'),scope)
