from kcat_prediction_slim.default_path import DefaultPath


def get_package_version(prefix_v: bool = True, use_underscore: bool = True):
    path = DefaultPath().project_root / 'pyproject.toml'
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version"):
                comment_removed = line.split("#")[0]
                version = comment_removed.split("=")[1].strip().strip('"')
    if prefix_v and use_underscore:
        return 'v' + version.replace('.', '_')
    elif prefix_v:
        return 'v' + version
    elif use_underscore:
        return version.replace('.', '_')
    return version


def get_package_major_version(prefix_v: bool = True):
    version_str = get_package_version(prefix_v=prefix_v, use_underscore=True)
    return version_str.split('_')[0]


if __name__ == '__main__':
    x = get_package_version()
    print(x)
