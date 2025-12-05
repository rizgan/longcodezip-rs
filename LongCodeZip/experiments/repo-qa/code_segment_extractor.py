import re
from typing import List, Dict, Optional
from loguru import logger
import json
import os

def extract_code_segments(code: str, language: str = "python") -> List[Dict]:
    """
    Break down code into a hierarchical structure based on language-specific patterns.
    Supports Python, C++, Java, TypeScript, Rust, and Go.
    
    Args:
        code: Original code string
        language: Programming language of the code (python, cpp, java, typescript, rust, go)
        
    Returns:
        List of code segments, each containing type, content, position, etc.
    """
    language = language.lower()
    
    # Language-specific patterns
    patterns = {
        "python": {
            "class": r"^class\s+(\w+)",
            "function": r"^def\s+(\w+)",
            "import": r"^(import|from)\s+",
            "comment": r"^#",
            "docstring": r'^("""|\'\'\')',
            "docstring_end": r'("""|\'\'\')$',
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith(":"),
            "block_end": lambda line, indent: len(line) - len(line.lstrip()) <= indent
        },
        "cpp": {
            "class": r"^(class|struct)\s+(\w+)",
            "function": r"^(void|int|bool|string|char|float|double|auto|template\s*<.*>)\s+(\w+)",
            "import": r"^#include\s+",
            "comment": r"^//|^/\*",
            "docstring": r"^/\*\*",
            "docstring_end": r"\*/$",
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith("{"),
            "block_end": lambda line, indent: line.rstrip() == "}" and len(line) - len(line.lstrip()) <= indent
        },
        "java": {
            "class": r"^(public|private|protected)?\s*(class|interface)\s+(\w+)",
            "function": r"^(public|private|protected)?\s*(void|int|boolean|String|char|float|double)\s+(\w+)",
            "import": r"^import\s+",
            "comment": r"^//|^/\*",
            "docstring": r"^/\*\*",
            "docstring_end": r"\*/$",
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith("{"),
            "block_end": lambda line, indent: line.rstrip() == "}" and len(line) - len(line.lstrip()) <= indent
        },
        "typescript": {
            "class": r"^(export\s+)?(class|interface)\s+(\w+)",
            "function": r"^(export\s+)?(function|const|let|var)\s+(\w+)\s*=",
            "import": r"^import\s+",
            "comment": r"^//|^/\*",
            "docstring": r"^/\*\*",
            "docstring_end": r"\*/$",
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith("{"),
            "block_end": lambda line, indent: line.rstrip() == "}" and len(line) - len(line.lstrip()) <= indent
        },
        "rust": {
            "class": r"^(pub\s+)?(struct|enum|trait)\s+(\w+)",
            "function": r"^(pub\s+)?fn\s+(\w+)",
            "import": r"^use\s+",
            "comment": r"^//|^/\*",
            "docstring": r"^//!|^/\*\*",
            "docstring_end": r"\*/$",
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith("{"),
            "block_end": lambda line, indent: line.rstrip() == "}" and len(line) - len(line.lstrip()) <= indent
        },
        "go": {
            "class": r"^type\s+(\w+)\s+(struct|interface)",
            "function": r"^func\s+(\w+)",
            "import": r"^import\s+",
            "comment": r"^//",
            "docstring": r"^//",
            "docstring_end": None,  # Go doesn't have multi-line docstrings
            "indent": lambda line: len(line) - len(line.lstrip()),
            "block_start": lambda line: line.rstrip().endswith("{"),
            "block_end": lambda line, indent: line.rstrip() == "}" and len(line) - len(line.lstrip()) <= indent
        }
    }
    
    if language not in patterns:
        raise ValueError(f"Unsupported language: {language}. Supported languages: {', '.join(patterns.keys())}")
    
    def get_token_length(text: str) -> int:
        """Simple approximation of token length by splitting by whitespace"""
        if not text:
            return 0
        return len(text.split())
    
    lines = code.split('\n')
    segments = []
    lang_patterns = patterns[language]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        indent_level = lang_patterns["indent"](lines[i])
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Process class/struct/enum/trait definitions
        class_match = re.match(lang_patterns["class"], line)
        if class_match and indent_level == 0:
            class_start = i
            class_name = class_match.group(1) if language == "python" else class_match.group(2)
            class_indent = indent_level
            
            # Save class header (signature and docstring) separately
            class_header_start = i
            
            # Skip to class body
            i += 1
            
            # Skip whitespace and comments to find the start of class body
            while i < len(lines) and (not lines[i].strip() or re.match(lang_patterns["comment"], lines[i].strip())):
                i += 1
            
            # Check for docstring
            if i < len(lines) and lang_patterns["docstring"] and re.match(lang_patterns["docstring"], lines[i].strip()):
                docstring_start = i
                i += 1
                # Find the end of the docstring
                while i < len(lines):
                    if lang_patterns["docstring_end"] and re.search(lang_patterns["docstring_end"], lines[i]):
                        i += 1
                        break
                    i += 1
            
            class_header_end = i
            class_header_code = '\n'.join(lines[class_header_start:class_header_end])
            
            # Continue processing the rest of the class body
            class_body_start = i
            
            # Extract methods/functions within the class
            while i < len(lines):
                if i >= len(lines) or (lines[i].strip() and lang_patterns["indent"](lines[i]) <= class_indent):
                    break
                
                line = lines[i].strip()
                current_indent = lang_patterns["indent"](lines[i])
                
                # Check for method/function definition
                method_indent = class_indent + (4 if language == "python" else 2)
                if re.match(lang_patterns["function"], line) and current_indent == method_indent:
                    method_start = i
                    method_name = re.match(lang_patterns["function"], line).group(1)
                    
                    # Find where method ends
                    i += 1
                    while i < len(lines):
                        if i < len(lines) and lines[i].strip() and lang_patterns["indent"](lines[i]) <= current_indent:
                            break
                        i += 1
                    
                    method_end = i
                    method_code = '\n'.join(lines[method_start:method_end])
                    
                    segments.append({
                        "type": "method",
                        "name": method_name,
                        "class_name": class_name,
                        "start_line": method_start,
                        "end_line": method_end,
                        "code": method_code,
                        "token_length": get_token_length(method_code),
                        "indent_level": current_indent
                    })
                    
                    continue
                else:
                    # Process non-method code (class attributes, etc.)
                    i += 1
            
            class_end = i
            class_code = '\n'.join(lines[class_start:class_end])
            
            # Add the class header segment
            segments.append({
                "type": "class_header",
                "name": class_name,
                "start_line": class_header_start,
                "end_line": class_header_end,
                "code": class_header_code,
                "token_length": get_token_length(class_header_code),
                "indent_level": class_indent
            })
            
            continue
        
        # Process function definitions
        func_match = re.match(lang_patterns["function"], line)
        if func_match and indent_level == 0:
            func_start = i
            func_name = func_match.group(1)
            func_indent = indent_level
            
            # Find the end of the function
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                current_indent = lang_patterns["indent"](lines[i])
                
                # If we hit another function or class at same or higher level, stop
                if (re.match(lang_patterns["function"], current_line) or re.match(lang_patterns["class"], current_line)) and current_indent <= func_indent:
                    break
                
                i += 1
            
            func_end = i
            func_code = '\n'.join(lines[func_start:func_end])
            
            segments.append({
                "type": "function",
                "name": func_name,
                "start_line": func_start,
                "end_line": func_end,
                "code": func_code,
                "token_length": get_token_length(func_code),
                "indent_level": 0
            })
            
            continue
        
        # Process imports
        if re.match(lang_patterns["import"], line) and indent_level == 0:
            import_start = i
            
            # Check if import statement spans multiple lines
            while i + 1 < len(lines) and (re.match(lang_patterns["import"], lines[i+1].strip()) or
                                         lines[i+1].lstrip().startswith('\\')):
                i += 1
            
            import_end = i + 1
            import_code = '\n'.join(lines[import_start:import_end])
            
            segments.append({
                "type": "import",
                "start_line": import_start,
                "end_line": import_end,
                "code": import_code,
                "token_length": get_token_length(import_code),
                "indent_level": 0
            })
            
            i += 1
            continue
        
        # Other top-level statements
        elif indent_level == 0:
            stmt_start = i
            
            # Find the end of the statement
            i += 1
            while i < len(lines) and (not lines[i].strip() or lang_patterns["indent"](lines[i]) > 0):
                i += 1
            
            stmt_end = i
            stmt_code = '\n'.join(lines[stmt_start:stmt_end])
            
            segments.append({
                "type": "statement",
                "start_line": stmt_start,
                "end_line": stmt_end,
                "code": stmt_code,
                "token_length": get_token_length(stmt_code),
                "indent_level": 0
            })
            
            continue
        
        # If nothing matched, move to next line
        i += 1
    
    return segments


# Example usage
if __name__ == "__main__":
    # Example Python code
    python_code = """
import os
import sys

class MyClass:
    \"\"\"This is a docstring.\"\"\"
    
    def __init__(self, name):
        self.name = name
    
    def my_method(self):
        print(f"Hello, {self.name}!")

def my_function():
    return "Hello, world!"

# This is a comment
x = 10
y = 20
z = x + y
    """
    
    # Example C++ code
    cpp_code = """
#include <iostream>
#include <string>

class MyClass {
public:
    MyClass(const std::string& name) : name_(name) {}
    
    void myMethod() {
        std::cout << "Hello, " << name_ << "!" << std::endl;
    }
    
private:
    std::string name_;
};

int myFunction() {
    return 42;
}

// This is a comment
int x = 10;
int y = 20;
int z = x + y;
    """
    
    # Test with Python
    python_segments = extract_code_segments(python_code, language="python")
    print(f"Python segments: {len(python_segments)}")
    
    # Test with C++
    cpp_segments = extract_code_segments(cpp_code, language="cpp")
    print(f"C++ segments: {len(cpp_segments)}") 