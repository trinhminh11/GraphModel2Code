from schemas import LibNode


torchlib_dict: dict[str, LibNode] = {}

def register_torchlib(name: str, lib_node: LibNode) -> None:
    torchlib_dict[name] = lib_node



