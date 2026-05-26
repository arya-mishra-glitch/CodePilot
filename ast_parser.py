"""
Layer 1 — AST Parser
Extracts meaningful code units from:
  Python      → functions, classes, methods (incl. nested)
  JavaScript  → functions, classes, methods, arrow functions  [+ Express routes]
  TypeScript  → same as JavaScript
  C           → functions, structs
  C++         → functions, classes, methods, structs
  Java        → classes, methods, constructors
  HTML        → inline <script> blocks (re-parsed as JS) + <style> blocks (re-parsed as CSS)
  CSS/SCSS    → rules, @media blocks, @keyframes blocks
 
Usage:
    python ast_parser.py <repo_path>            # tree view of all supported files
    python ast_parser.py <repo_path> --json     # JSON output (flat list)
    python ast_parser.py <file.py>              # single file
    python ast_parser.py <repo_path> --no-code  # metadata only, no source
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tsts
import tree_sitter_html as tshtml
import tree_sitter_css as tscss
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava

from tree_sitter import Language, Parser, Node


#Data Model

#---------------------------------------------------------------------------------------------


#dataclass is a Python decorator that automatically generates special methods for a class based on special fields

@dataclass

class CodeUnit:

    """ This is a single standard unit for all languages. The kind values may include:
        
        Python/Js/C/C++/Java        -> function | method | class | struct | constructor | arrow_function
        Express (JS)                -> route
        HTML                        -> script_block | style_block
        CSS                         -> rule | media_rule | keyframes_rule
    
        
    """


    kind: str  
    name: str
    file: str
    start_line: int
    end_line: int
    parent: Optional[str]       #enclosing class/ selector/ route prefix
    
    code: str                   # the actual sourcce code text corresponding to that extracted unit
    language: str
    children: list = field(default_factory=list)    #When creating a new object, call list() to generate the default value.

    def display_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


#------------------------------------------------------------------------------------------------


#Language Registry


PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
HTML_LANGUAGE = Language(tshtml.language())
CSS_LANGUAGE = Language(tscss.language())
C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())
JAVA_LANGUAGE = Language(tsjava.language())



LANGUAGE_MAP: dict[str, tuple[str, Language]] = {
    ".py":      ("python",          PY_LANGUAGE),
    ".js":      ("javascript",      JS_LANGUAGE),
    ".mjs":     ("javascript",      JS_LANGUAGE),
    ".cjs":     ("javascript",      JS_LANGUAGE),
    ".jsx":     ("javascript",      JS_LANGUAGE),
    ".ts":      ("typescript",      TS_LANGUAGE),
    ".tsx":     ("typescript",      TSX_LANGUAGE),
    ".html":    ("html",            HTML_LANGUAGE),
    ".htm":     ("html",            HTML_LANGUAGE),
    ".css":     ("css",             CSS_LANGUAGE),
    ".scss":    ("css",             CSS_LANGUAGE),
    ".c":       ("c",               C_LANGUAGE),
    ".h":       ("c",               C_LANGUAGE),
    ".cpp":     ("cpp",             CPP_LANGUAGE),
    ".cc":      ("cpp",             CPP_LANGUAGE),
    ".cxx":     ("cpp",             CPP_LANGUAGE),
    ".hpp":     ("cpp",             CPP_LANGUAGE),
    ".java":    ("java",            JAVA_LANGUAGE),
    
}


#-----------------------------------------------------------------------------------------------------------------------------

#Helper Functions


#Function to extract the raw source text corresponding to an AST node

def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


#Function to get the first name identifier

def _first_id(node: Node, source: bytes, 
              types=("identifier", "field_identifier",
                     "property_identifier", "type_identifier")) -> Optional[str]:
    for child in node.children:
        if child.type in types:
            return _text(child, source)
        
    return None

# Create a CodeUnit from an AST node

def _make(kind, name, node, source, file_path, language, parent=None) -> CodeUnit:
    return CodeUnit(
        kind= kind,
        name = name or "<anonymous>",
        file=file_path,
        start_line= node.start_point[0] +1, #because it's 0-index
        end_line= node.end_point[0] + 1,
        parent=parent,
        code= _text(node, source),          #extract the raw source text/code
        language= language,
    )


#-------------------------------------------------------------------------------------------------------------------------------
#Extractor Functions
#-------------------------------------------------------------------------------------------------------------------------------



#Python

def _extract_python(node: Node, source: bytes, file_path: str, parent_class: Optional[str] = None) -> list[CodeUnit]:
    units=[]
    for child in node.children:
        if child.type == "class_definition":
            name = _first_id(child, source)
            u = _make("class", name, child, source, file_path, "python", parent_class)
            u.children = _extract_python(child, source, file_path, name)
            units.append(u)

        elif child.type == "decorated_definition":
            inner= next((c for c in child.children 
                        if c.type in ("function_definition", "class_definition")), None)  
            if inner:
                name = _first_id(inner, source)
                if inner.type == "class_definition":
                    u = _make("class", name, child, source, file_path, "python", parent_class)
                    u.children = _extract_python(inner, source, file_path, name)
                else:
                    kind = "method" if parent_class else "function"
                    u = _make(kind, name, child, source, file_path, "python", parent_class)
                    u.children = _extract_python(inner, source, file_path, name)
                units.append(u)

        elif child.type == "function_definition":
            name = _first_id(child, source)
            kind = "method" if parent_class else "function" #if its a class, treat is as a method, not function
            u = _make(kind, name, child, source, file_path, "python", parent_class)
            u.children = _extract_python(child, source, file_path, name)
            units.append(u)
        else:
            units.extend(_extract_python(child, source, file_path, parent_class))

    return units

#----------------------------------------------------------------------------------------------------------------------------------

#JavaScript / TypeScript/ Express

_EXPRESS_OBJECTS = {"app", "router", "Router"}
_EXPRESS_VERBS = {"get", "post", "put", "patch", "delete", "use", "all"}


def _member_expr_last_prop(node: Node, source: bytes) -> Optional[str]:
    """Extract the rightmost property name from a member_expression.
    
    exports.login           -> 'login' (not 'exports')
    module.exports.getUser  -> 'getUser' (not 'module)
    plainIdentifier         -> 'plainIdentifier'
    """
    if node.type == "member_expression":
        for child in reversed(node.children):
            if child.type == "property_identifier":
                return _text(child, source)
            
    #fallback: plain identifier (e.g. bare `someVar = asyn () => {}` )
    return _first_id(node, source)




"""AST for routes roughly:

call_expression
├── member_expression (app.get)
└── arguments """



def _express_route(call_node: Node, source:bytes) -> Optional[tuple[str, str]]:
    """Return (HTTP_VERB, '/path') if this looks like app.get('/path', ...) else None"""

    member = next((c for c in call_node.children if c.type == "member_expression"), None)
    if not member:
        return None
    
    parts = _text(member, source).split(".")        # if app.get then parts = ["app", "get"]

    if len(parts) == 2 and parts[0] in _EXPRESS_OBJECTS and parts[1] in _EXPRESS_VERBS:
        args = next((c for c in call_node.children if c.type == "arguments"), None)
        if args:
            for child in args.children:
                if child.type in ("string", "template_string"):
                    path = _text(child, source).strip("'\"`")
                    return parts[1].upper(), path
        return None
    
def _extract_javascript(node: Node, source: bytes, file_path: str, 
                        parent_class: Optional[str]= None) -> list[CodeUnit]:
    units = []
    for child in node.children:

        #Express route detection
        if child.type== "expression_statement":
            call = next((c for c in child.children if c.type == "call_expression"), None)
            if call:
                route = _express_route(call, source)
                if route:
                    method, path = route
                    u= _make("route", f"{method} {path}", child, source,
                            file_path, "javascript")
                    units.append(u)
                    continue

            #handle module.exports = async function / assignment with function
            assign = next((c for c in child.children if c.type == "assignment_expression"), None)
            if assign: 
                right = assign.children[-1] if assign.children else None        #last child is function 
                if right and right.type in ("arrow_function", "function_expression"):
                    left = assign.children[0] if assign.children else None
                    name = _member_expr_last_prop(left, source) if left else None
                    #eg: exports.login = async () => {} helpler extracts login which is correct
                    #but module.exports = async function login() {} would extract exports, which is wrong. login should be the name. so,
                                        # if left is `module.exports` (bare export), _member_expr_last_prop
                    # returns "exports" — fall back to the function's own identifier instead
                    if name == "exports":
                        name = _first_id(right, source)
                    if name:
                        kind = "method" if parent_class else "arrow_function"
                        # use child (expression_statement) so code includes "exports.X = ..."
                        u= _make(kind, name, child, source, file_path, "javascript", parent_class)
                        u.children = _extract_javascript(right, source, file_path, parent_class)
                        units.append(u)
                        continue

                #fallback: recurse (e.g. module.exports = { ... }  object)
                units.extend(_extract_javascript(assign, source, file_path, parent_class))
                continue


        if child.type in ("class_declaration", "class"):
            name = _first_id(child, source)
            u = _make("class", name, child, source, file_path, "javascript", parent_class)
            u.children = _extract_javascript(child, source, file_path, name)
            units.append(u)

        elif child.type == "method_definition":
            name = _first_id(child, source, 
                             types = ("identifier", "property_identifier", 
                                      "private_property_identifier"))
            u = _make("method", name, child, source, file_path, "javascript", parent_class)
            units.append(u)
        
        elif child.type in ("function_declaration", "generator_function_declaration"):  
            name = _first_id(child, source)
            kind = "method" if parent_class else "function"
            u = _make(kind, name, child, source, file_path, "javascript", parent_class)
            units.append(u)

        elif child.type in ("lexical_declaration", "variable_declaration"):
            found_callable = False
            for decl in child.children:
                if decl.type == "variable_declarator":  
                    var_name = _first_id(decl, source)
                    val =  next((c for c in decl.children
                                if c.type in ("arrow_function", "function_expression")), None)
                    
                    if val and var_name:
                        kind = "method" if parent_class else "arrow_function"
                        u= _make(kind, var_name, decl, source, file_path, 
                                 "javascript", parent_class)
                        u.children = _extract_javascript(val, source, file_path, parent_class)
                        units.append(u)
                        found_callable = True
            #if no function in this declaration, still recurse (eg: deconstructed exports)

            if not found_callable:
                units.extend(_extract_javascript(child, source, file_path, parent_class))

        #handle: export const fn = .../ export function fn / export default function 
        elif child.type == "export_statement":
            units.extend(_extract_javascript(child, source, file_path, parent_class))


        # module.exports = {
        #    login: async () => {}
        # }
        #AST stores it as
        # object
        # └── pair
        #     └── arrow_function


        #handle: object containing methods, e.g module exports = {getUser , ...}
        elif child.type == "object":
            units.extend(_extract_javascript(child, source, file_path, parent_class))

         # handle: property in object: key: async function() {}
        elif child.type == "pair":                                                                  
            val = next((c for c in child.children
                        if c.type in ("arrow_function", "function_expression",
                                      "function")), None)
            if val:
                key = next((c for c in child.children
                            if c.type in ("property_identifier", "identifier",                                                                      
                                          "string")), None)
                name = _text(key, source).strip("'\"") if key else None
                if name:
                    kind = "method" if parent_class else "arrow_function"
                    u = _make(kind, name, val, source, file_path, "javascript", parent_class)
                    u.children = _extract_javascript(val, source, file_path, parent_class)
                    units.append(u)
            else:
                units.extend(_extract_javascript(child, source, file_path, parent_class))

         # handle: shorthand_property_identifier in exports object: { myFunc }
        elif child.type == "assignment_expression":
            # e.g. module.exports = { ... } or exports.fn = async () => {}
            right = child.children[-1] if child.children else None
            if right and right.type in ("arrow_function", "function_expression"):
                left = child.children[0] if child.children else None
                name = _member_expr_last_prop(left, source) if left else None
                if name == "exports":
                    name = _first_id(right, source)
                if name:
                    kind = "method" if parent_class else "arrow_function"
                    u = _make(kind, name, right, source, file_path, "javascript", parent_class)
                    u.children = _extract_javascript(right, source, file_path, parent_class)
                    units.append(u)
            else:
                units.extend(_extract_javascript(child, source, file_path, parent_class))


        else:
            units. extend(_extract_javascript(child, source, file_path, parent_class))

    return units

#-----------------------------------------------------------------------------------------------------------------------------------

# C

# Because C tree sitter roughly
# function_definition
# ├── primitive_type (int)
# ├── function_declarator
# │   ├── identifier (add)
# │   └── parameter_list
# └── compound_statement

def _c_func_name(node: Node, source: bytes) -> Optional[str]:
    for child in node.children:
        if child.type == "function_declarator":
            return _first_id(child, source)
    return None

def _extract_c(node: Node, source:bytes, file_path:str, parent: Optional[str] = None) -> list[CodeUnit]:
    units = []

    for child in node.children:
        if child.type == "function_definition":
            name = _c_func_name(child, source)
            units.append(_make("function", name, child, source, file_path, "c", parent))

        elif child.type in ("struct_specifier", "union_specifier"):
            name=_first_id(child, source, types=("type_identifier", "identifier"))
            units.append(_make("struct", name, child, source, file_path, "c", parent))
        else:
            units.extend(_extract_c(child, source, file_path, parent))
    return units  
#-----------------------------------------------------------------------------------------------------------------------------------------

#C++

def _cpp_method_name(node: Node, source: bytes) -> Optional[str]:
    for child in node.children:
        if child.type == "function_declarator":
            return _first_id(child, source, 
                             types=("field_identifier", "identifier",
                                    "destructor_name", "operator_name"))
    return None

def _extract_cpp(node: Node, source: bytes, file_path:str,
                 parent_class: Optional[str] = None) -> list[CodeUnit]:
    units=[]
    for child in node.children:
        if child.type=="class_specifier":
            name = _first_id(child, source, types=("type_identifier", "identifier"))
            u= _make("class", name, child, source, file_path, "cpp", parent_class)
            u.children = _extract_cpp(child, source, file_path, name)
            units.append(u)
        
        elif child.type in ("struct_specifier", "union_specifier"):
            name = _first_id(child, source, types=("type_identifier", "identifier"))
            u = _make("struct", name, child, source, file_path, "cpp", parent_class)
            u.children = _extract_cpp(child, source, file_path, name)
            units.append(u)
        elif child.type == "function_definition":
            name= _cpp_method_name(child, source)
            kind= "method" if parent_class else "function"
            units.append(_make(kind, name, child, source, file_path, "cpp", parent_class))
        else:
            units.extend(_extract_cpp(child, source, file_path, parent_class))
    return units  
    
#-------------------------------------------------------------------------------------------------------------------
#Java

def _extract_java(node: Node, source: bytes, file_path: str, 
                  parent_class: Optional[str] = None) -> list[CodeUnit]:
    units=[]
    for child in node.children:
        if child.type == "class_declaration":
            name = _first_id(child, source)
            u = _make("class", name, child, source, file_path, "java", parent_class)
            u.children = _extract_java(child, source, file_path, name)
            units.append(u)
        elif child.type == "method_declaration":
            name = _first_id(child, source)
            units.append(_make("method", name, child, source, file_path, "java", parent_class))  # fix 6: added "java" and parent_class
        elif child.type == "constructor_declaration":
            name = _first_id(child, source)
            units.append(_make("constructor", name, child, source, file_path, "java", parent_class))
        else :
            units.extend(_extract_java(child, source, file_path, parent_class))
    return units

#-----------------------------------------------------------------------------------------------------------------------------------
#CSS 
# rule_set
# ├── selectors
# │   └── .container
# └── block


#for media statements, the AST looks like this
# eg: @media screen and (max-width: 600px) {
#     .container {
#         display: none;
#     }
# }

# media_statement
# ├── @media
# ├── screen
# ├── and
# ├── (max-width: 600px)
# └── block (actual CSS rules)


def _css_selector(node: Node, source: bytes) -> str:
    sel = next((c for c in node.children if c.type == "selectors"), None)
    return _text(sel, source).strip() if sel else "<unknown>"

def _extract_css (node: Node, source: bytes, file_path: str,
                  parent: Optional[str]= None) -> list[CodeUnit]:
    units = []
    for child in node.children:
        if child.type == "rule_set":    #rule_set is the AST node type for a normal CSS rule
            name = _css_selector(child, source)
            units.append(_make("rule", name, child, source, file_path, "css", parent))

        elif child.type == "media_statement":
            #collect everything between @media and { as query string
            parts=[]
            for c in child.children:
                if c.type == "block":
                    break
                if c.type != "@media":
                    parts.append(_text(c,source).strip())
            query = " ".join(p for p in parts if p)
            name = f"@media {query}"
            u= _make("media_rule", name, child, source, file_path, "css", parent)
            block = next((c for c in child.children if c.type=="block"), None)

            if block: 
                u.children = _extract_css(block, source, file_path, name)
            units.append(u)

        elif child.type == "keyframes_statement":
            kname = next((c for c in child.children if c.type == "keyframes_name"), None)
            name = f"@keyframes {_text(kname, source)}" if kname else "@keyframes"  # fix 7: - changed to =
            units.append(_make("keyframes_rule", name, child, source, file_path, "css", parent))
        else:
            units.extend(_extract_css(child, source, file_path, parent))
    return units  
    
#------------------------------------------------------------------------------------------------------------------------------
#HTML

_JS_PARSER = Parser(JS_LANGUAGE)
_CSS_PARSER = Parser(CSS_LANGUAGE)

def _extract_html(node: Node, source: bytes, file_path: str) -> list[CodeUnit]:
    units=[]
    for child in node.children: 
        if child.type == "script_element":
            raw = next((c for c in child.children if c.type == "raw_text"), None)
            if raw:
                js_src = source[raw.start_byte:raw.end_byte]
                inner = _extract_javascript(_JS_PARSER.parse(js_src).root_node, 
                                            js_src, file_path)      ## Parse extracted JS source into an AST and pass its root node to the JS extractor
                if inner:
                    block = _make("script_block", 
                                  f"<script> line {raw.start_point[0]+1}",
                                  raw, source, file_path, "javascript")  # fix 9: "css" -> "javascript"
                    block.children = inner
                    units.append(block)

        elif child.type == "style_element":  # fix 10: wrong indent + "Style_element" -> "style_element"
            raw= next((c for c in child.children if c.type == "raw_text"), None)
            if raw:
                css_src = source[raw.start_byte: raw.end_byte]  # fix 11: start_type -> start_byte
                inner = _extract_css(_CSS_PARSER.parse(css_src).root_node,
                                     css_src, file_path)
                if inner: 
                    block = _make("style_block", 
                                  f"<style> line {raw.start_point[0]+1}",
                                  raw, source, file_path, "css")
                    block.children = inner
                    units.append(block)
        
        else:
            units.extend(_extract_html(child, source, file_path))
    return units 
    
#------------------------------------------------------------------------------------------------------------------
#dispatch table

_EXTRACTORS = {
    "python":       _extract_python,
    "javascript":   _extract_javascript,
    "typescript":   _extract_javascript,
    "c":            _extract_c,
    "cpp":          _extract_cpp,
    "java":         _extract_java,
    "css":          _extract_css,
    "html":         lambda n, s, f: _extract_html(n,s,f),
}

#---------------------------------------------------------------------------------------------------------------------------
#Public API

def parse_file(file_path: str) -> list[CodeUnit]:
    ext = Path(file_path).suffix.lower()
    if ext not in LANGUAGE_MAP:
        return []
    lang_name, language = LANGUAGE_MAP[ext]
    parser = Parser(language)
    with open(file_path, "rb") as f:
        source = f.read()
    tree = parser.parse(source)
    return _EXTRACTORS[lang_name](tree.root_node, source, file_path)

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", "out", ".idea", ".vscode",
}


def parse_repo(repo_path:str, 
               extensions: Optional[list[str]] = None) -> list[CodeUnit]:
    supported = set(extensions or LANGUAGE_MAP.keys())
    all_units: list[CodeUnit] = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext in supported:
                full_path = os.path.join(dirpath, fname)
                try:
                    all_units.extend(parse_file(full_path))
                except Exception as e:
                    print(f" [warn] {full_path}: {e}", file = sys.stderr)
    return all_units


def _flatten(units: list[CodeUnit]) -> list[CodeUnit]:
    result = []
    for u in units:
        result.append(u)
        result.extend(_flatten(u.children))  
    return result

def units_to_records(units: list[CodeUnit]) -> list[dict]:
    """Flat JSON - serialisable list - input format for Layer 2."""
    rows = []
    for u in _flatten(units):
        d = asdict(u)
        d.pop("children")
        rows.append(d)
    return rows



#----------------------------------------------------------------------------------------------------------------------
#Pretty Print

_ICONS = {
    "class": "🔷", "struct": "🔶", "function": "🟢", "method": "🔵",
    "constructor": "🟣", "arrow_function": "🟩", "route": "🌐",
    "rule": "🎨", "media_rule": "📐", "keyframes_rule": "✨",
    "script_block": "📜", "style_block": "🖌️",
}

def _print_tree(units: list[CodeUnit], indent:int =0)-> None:
    pad = "     " * indent
    for u in units:
        icon = _ICONS.get(u.kind, "❓")
        loc = f"{Path(u.file)}:{u.start_line}-{u.end_line}"
        print(f"{pad}{icon} {u.display_name()}  ({loc})")
        if u.children:  
            _print_tree(u.children, indent+1)

def print_summary(units: list[CodeUnit], repo_path:str) -> None:
    by_file = {}
    for u in units:
        by_file.setdefault(u.file, []).append(u)  

    by_kind: dict[str, int] = {}
    for u in _flatten(units):
         by_kind[u.kind] = by_kind.get(u.kind, 0) + 1
    summary = ", ".join(f"{v} {k}(s)" for k,v in sorted(by_kind.items())) or "none"

    print(f"\n{'='*62}")
    print(f"  Layer 1 — AST Parse Results")
    print(f"  Path  : {repo_path}")
    print(f"  Files : {len(by_file)}")
    print(f"  Units : {summary}")
    print(f"{'='*62}\n")

    for file_path, file_units in sorted(by_file.items()):
        rel  = os.path.relpath(file_path, repo_path)
        lang = LANGUAGE_MAP.get(Path(file_path).suffix.lower(), ("?",))[0]
        print(f"📄 {rel}  [{lang}]")
        _print_tree(file_units, indent=1)
        print()

#------------------------------------------------------------------------------------------------------------------------
#CLI

def main():
    ap = argparse.ArgumentParser(
        description = "Layer 1 - Multi-language AST code parser",
        epilog = "Supported: .py  .js .ts .jsx .tsx  .html .css .scss  .c .h .cpp .hpp  .java"
    )
    ap.add_argument("path",      help="Repo directory or single source file")
    ap.add_argument("--json",    action="store_true", help="Flat JSON array output")
    ap.add_argument("--no-code", action="store_true", help="Strip source code from output")
    args = ap.parse_args()

    target = os.path.abspath(args.path)     #Convert the user-provided path into a full absolute path


    if os.path.isfile(target):
        units     = parse_file(target)
        repo_root = os.path.dirname(target)
    elif os.path.isdir(target):
        units     = parse_repo(target)
        repo_root = target
    else:
        print(f"Error: '{target}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.no_code:
        for u in _flatten(units):
            u.code = ""
 
    if args.json:
        print(json.dumps(units_to_records(units), indent=2))
    else:
        print_summary(units, repo_root)

if __name__ == "__main__":
    main()