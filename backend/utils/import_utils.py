from typing import Sequence

class DependencyNode:
    def __init__(self, name: str, alias: str | None = None):
        self.name = name
        self.alias = alias    # for import ...name as alias

        self.is_import = False  # if need to import this node
        self.children: dict[str, DependencyNode] = {}


    def add_child(self, child: "DependencyNode") -> None:
        self.children[child.name] = child

    def contains(self, dependency_name: str) -> bool:
        if dependency_name in self.children:
            return True
        return False

    def not_contains(self, dependency_name: str) -> bool:
        return not self.contains(dependency_name)


    def __repr__(self):
        return f"DependencyNode(name={self.name}, alias={self.alias}, is_import={self.is_import}, children={[child for child in self.children]})"

class DependencyTree:
    """
    DependencyTree is a tree that represents the dependencies of the nodes.
    the root of the tree is just a placeholder node that represents the root of the dependencies.
    all the children of the root are the libraries that are used in the dependencies.
    all the leaf nodes are the nodes that are used in the dependencies.
    """
    def __init__(self):
        self.root = DependencyNode("__root__")

    def check_dependency(self, dependency: str) -> bool:
        name = dependency.strip()
        if " as " in name:
            name, alias = name.split(" as ")
            name = name.strip()
            alias = alias.strip()
        else:
            alias = None

        if "." in name:
            raise ValueError(f"Invalid dependency: {dependency}, cannot contain '.' in the name")

        if " " in name or (alias is not None and " " in alias):
            raise ValueError(f"Invalid dependency: {dependency}, cannot contain any spaces in the name")

        return name, alias


    def add_dependency(self, dependencies: Sequence[str]) -> "DependencyTree":
        current_node = self.root
        for dependency in dependencies[:-1]:
            name, alias = self.check_dependency(dependency)
            if current_node.not_contains(name):
                current_node.add_child(DependencyNode(name, alias))
            current_node = current_node.children[name]

        # in the final dependency, we need to check the alias is the same as the previous dependencies or it's a new alias
        name, alias = self.check_dependency(dependencies[-1])
        if current_node.not_contains(name):
            current_node.add_child(DependencyNode(name, alias))
        else:
            if alias is not None:
                if current_node.children[name].alias is None:
                    current_node.children[name].alias = alias
                elif current_node.children[name].alias != alias:
                    raise ValueError(f"dependency {dependency} has different alias, current alias is {current_node.children[name].alias} but got {alias}")
        current_node = current_node.children[name]
        current_node.is_import = True

        return self

    def add_dependencies(self, dependencies: Sequence[Sequence[str]]) -> "DependencyTree":
        for dependency in dependencies:
            self.add_dependency(dependency)

        return self

    def dfs(self, node: DependencyNode) -> None:
        print(node)
        for child in node.children.values():
            self.dfs(child)

    def get_lib(self):
        "function to generate required libraries"

        return [node for node in self.root.children]

    def simple_import_code(self):
        "function to generate the import code for the dependencies"

        ret: list[list[str]] = []

        def _dfs(node: DependencyNode, dependencies: list[str]) -> None:
            current_dependencies = dependencies.copy()
            current_dependencies.append(node.name)
            for child in node.children.values():
                _dfs(child, current_dependencies)

            if node.is_import:
                ret_dependencies = current_dependencies.copy()
                if node.alias is not None:
                    ret_dependencies[-1] = f"{ret_dependencies[-1]} as {node.alias}"
                ret.append(ret_dependencies)


        for child in self.root.children.values():
            dependencies = []
            _dfs(child, dependencies)

        return ret

    def generate_import_code(self, add_dot: bool = False):
        import_list = self.simple_import_code()

        ret = ""
        for dependencies in import_list:
            if add_dot:
                if len(dependencies) == 1:
                    ret += "from . import " + dependencies[0] + "\n"
                else:
                    ret += "from ." + ".".join(dependencies[:-1]) + " import " + dependencies[-1] + "\n"
            else:
                if len(dependencies) == 1:
                    ret += "import " + dependencies[0] + "\n"
                else:
                    ret += "from " + ".".join(dependencies[:-1]) + " import " + dependencies[-1] + "\n"

        return ret

def get_dependencies_str(dependencies: Sequence[Sequence[str]], add_dot: bool = False) -> str:
    tree = DependencyTree().add_dependencies(dependencies)
    return tree.generate_import_code(add_dot)

def main():
    test = [["torch", "nn"], ["torch", "Tensor"], ["torch", "nn", "functional as F"], ["torch", "nn", "Module"]]
    # expected:
    # from torch import nn, Tensor
    # from torch.nn import Module, functional as F
    # currently:
    # from torch.nn import functional as F
    # from torch.nn import Module
    # from torch import nn
    # from torch import Tensor
    tree = DependencyTree()
    tree.add_dependencies(test)

    print(tree.generate_import_code())


if __name__ == "__main__":
    main()
