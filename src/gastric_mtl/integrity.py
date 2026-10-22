"""Version-stable fingerprints for unchanged numerical Python definitions."""
import ast
import hashlib
from pathlib import Path


def definition_fingerprints(path):
    tree = ast.parse(Path(path).read_text())
    # Python 3.12 adds an empty type_params field to ordinary functions/classes.
    # Exclude that empty field so 3.9 and newer compare the same definitions.
    # Nonempty generic parameters remain significant and are never suppressed.
    for node in ast.walk(tree):
        if 'type_params' in node._fields and not node.type_params:
            node._fields = tuple(field for field in node._fields if field != 'type_params')
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
