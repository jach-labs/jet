"""Rebuild additional diagnostic families with the existing official normalizers.

This is a partial reconstruction, never a Decision Index 0.2 leaderboard score.
"""
import json,shutil
from pathlib import Path
from decision_index.suite.build.acquire import acquire
from decision_index.suite.build.layout import Layout
from decision_index.suite.build.rebuild import main as rebuild

def main():
 work=Path('artifacts/decision-index/v5-rebuild');layout=Layout(work)
 # The upstream knowledge builder needs all of these even to select HellaSwag.
 acquire(layout,[5,12,24,26,27,28,29])
 # The installed acquirer stages raw/ at root; normalizers read suite/raw/.
 for name in ['clinc150','anli']:
  shutil.copytree(layout.root/'raw'/name,layout.raw/name,dirs_exist_ok=True)
 result=rebuild(work,only=[5,12,29],skip_download=True)
 out=Path(result['out'])
 shutil.copy2('artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt/excluded-questions.json',out/'excluded-questions.json')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
