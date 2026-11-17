"""Generate machine-local MuJoCo paths without changing model parameters."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check resources without writing a local scene')
    args = parser.parse_args()
    path = ROOT / 'assets/piper/scene.template.xml'
    tree = ET.parse(path)
    count = 0
    for mesh in tree.getroot().findall('./asset/mesh'):
        source = (ROOT / mesh.attrib['file']).resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Missing mesh: {source}')
        # Keep support for the original Windows Unicode path workaround.
        from sys import path as sys_path
        if str(ROOT / 'src') not in sys_path:
            sys_path.insert(0, str(ROOT / 'src'))
        # No MuJoCo dependency is needed just to check/prepare file paths.
        # Windows short paths are an optional rendering compatibility aid.
        value = str(source)
        import os
        if os.name == 'nt':
            import ctypes
            buffer = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.kernel32.GetShortPathNameW(value, buffer, len(buffer)):
                value = buffer.value
        mesh.set('file', value)
        count += 1
    if count != 10:
        raise ValueError(f'Expected 10 robot meshes, found {count}')
    ui = ROOT / 'assets/ui/task_console.html'
    if not ui.is_file():
        raise FileNotFoundError(ui)
    target = ROOT / 'assets/piper/scene.xml'
    if not args.check:
        ET.indent(tree)
        tree.write(target, encoding='utf-8', xml_declaration=True)
    print(f'Robot meshes: {count}; UI: present; mode: {"check" if args.check else "prepare"}')
    if not args.check:
        print(f'Local scene: {target}')

if __name__ == '__main__':
    main()
