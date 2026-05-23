from tree_sitter import Language, Parser
import tree_sitter_python as tspython

#importing Language for the syntax of the language
#Parser is the mechanism that builds the AST tree

PY_LANGUAGE = Language(tspython.language())
#PY_LANGUAGE now contains the syntax and rules of Python
parser= Parser(PY_LANGUAGE)
# Create a parser configured to understand Python syntax



def parse_file(filepath: str) -> None:
    #open read only in binary mode
    with open(filepath, "rb") as f:
        source = f.read()

    tree = parser.parse(source)
    #Parser parses the source code
    root=tree.root_node

    functions = extract_functions(root, source)
    for name, start_line in functions:
        print(f"Line {start_line:>4} : {name}()")


def extract_functions(node, source: bytes) -> list[tuple[str, int]]:
    results=[]

#     because internal structure looks like this
# function_definition
# ├── def 
# ├── hello (name)
# ├── parameters
# └── body


    if node.type == "function_definition":

        name_node = node.child_by_field_name("name")

        if name_node:
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8") #convert to UTF coding from binary
            line = node.start_point[0]+1 #tree sitter is 0-indexed
            results.append((name, line))

    for child in node.children:
        results.extend(extract_functions(child,source))

    return results

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv)>1 else __file__
    print(f"\nFunctions in: {target}\n")
    parse_file(target)
