from pathlib import Path
import shutil
from datetime import datetime

root = Path(__file__).resolve().parents[1]
installed = Path('C:/Users/shado/AppData/Roaming/Blender Foundation/Blender/5.2/scripts/addons/cobra-tools-master')
init = installed / '__init__.py'
text = init.read_text(encoding='utf-8')
import_anchor = '\t\tfrom .plugin import addon_updater_ops\n'
class_anchor = '\t\tclasses = (\n'
assert import_anchor in text and class_anchor in text
backup = root / 'leaf_panel_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
backup.mkdir(parents=True)
shutil.copy2(init, backup / '__init__.py')
module = installed / 'plugin/leaf_bones.py'
if module.exists():
    shutil.copy2(module, backup / 'leaf_bones.py')
if 'from .plugin.leaf_bones import LEAF_CLASSES' not in text:
    text = text.replace(import_anchor, import_anchor + '\t\tfrom .plugin.leaf_bones import LEAF_CLASSES\n', 1)
    text = text.replace(class_anchor, class_anchor + '\t\t\t*LEAF_CLASSES,\n', 1)
shutil.copy2(root / 'plugin/leaf_bones.py', module)
init.write_text(text, encoding='utf-8')
print('Installed leaf panel; existing addon changes retained. Backup:', backup)
