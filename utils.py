from __future__ import annotations

def replace_extension(filename: str, new_extension: str):
    parts = filename.split('.')

    if len(parts) > 1:
        del parts[-1]

    parts.append(new_extension)

    return '.'.join(parts)
