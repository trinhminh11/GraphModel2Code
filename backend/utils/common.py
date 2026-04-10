def get_dependencies_str(dependencies: set[str]) -> str:
    # This function should involve like: {"from a import b", "from a import c"} can turn into "from a import b, c"
    # but for now, just return the dependencies as a string
    dependencies = sorted(dependencies)

    return "\n".join(dependencies)
