import ast
import re


class FunctionAnalyzer():
  def __init__(self, is_strict=False):
    self.is_strict = is_strict

  def get_function_name(self,code):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
          if isinstance(node, ast.FunctionDef):
            return node.name
    except Exception as e:
        if self.is_strict:
            raise
        return ""
        
  def get_function_def_line(self,code):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
          if isinstance(node, ast.FunctionDef):
            return node.lineno - 1
        return -1 
    except Exception as e:
        if self.is_strict:
            raise
        return -1
  
  def get_docstring(self,code):
    try:
      tree = ast.parse(code)
      for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
          docstring = ast.get_docstring(node)
          if docstring:
            return docstring
      return None
    except SyntaxError as e:
      if self.is_strict:
        raise
      return None

  def remove_docstring_from_function(self,func_str):
    # Parse the function string into an AST node
    try:
        tree = ast.parse(func_str)
        
        # Find the function definition in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    # Remove the first node in the body if it's a docstring
                    node.body.pop(0)
                break
    
        # Convert the modified AST back to source code
        return ast.unparse(tree)
    except SyntaxError as e:
      if self.is_strict:
        raise
      return func_str
  def remove_leading_whitespace_for_def(self,code):
        lines = code.split('\n')
        updated_lines = []
        for line in lines:
            stripped_line = line.lstrip()
            if stripped_line.startswith('def '):
                updated_lines.append(stripped_line)
            else:
                updated_lines.append(line)
        return '\n'.join(updated_lines)
      
  def get_function_blocks(self,code):
    try:
        tree = ast.parse(self.remove_leading_whitespace_for_def(code))
        function_blocks = []
        for node in ast.walk(tree):
          if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')]) - 1
            function_block = '\n'.join(code.splitlines()[start_line:end_line + 1])
            function_blocks.append(function_block)
        return function_blocks 
    
    except Exception as e:
        if self.is_strict:
            raise
        return []
        
  def get_code_blocks(self,code):
      try:
          tree = ast.parse(code)
          code_blocks = []
          block_info = []
        
          # Traverse the AST nodes
          for node in ast.walk(tree):
              if isinstance(node, (ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    start_line = node.lineno
                    end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')])
                    # Extract the code block from the original code
                    block = '\n'.join(code.splitlines()[start_line - 1:end_line])
                    # block = {'block_content':block,'start_line':start_line - 1,'end_line':end_line}
                    code_blocks.append(block)
                    # if isinstance(node, (ast.If, ast.For, ast.While, ast.With,ast.Try)):
                    line = code.splitlines()[start_line]
                    indent_level = len(line) - len(line.lstrip())
                    block_info.append((start_line - 1,end_line,indent_level,type(node)))
              elif isinstance(node, (ast.FunctionDef)):
                    start_line = node.lineno
                    end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')])
                    # Extract the code block from the original code
                    block = '\n'.join(code.splitlines()[start_line:end_line])
                    # block = {'block_content':block,'start_line':start_line - 1,'end_line':end_line}
                    code_blocks.append(block)
                    # if isinstance(node, (ast.If, ast.For, ast.While, ast.With,ast.Try)):
                    line = code.splitlines()[start_line]
                    indent_level = len(line) - len(line.lstrip())
                    block_info.append((start_line,end_line,indent_level,type(node)))               
          return code_blocks, block_info
      except Exception as e:
          if self.is_strict:
              raise
          return [],[]  

  def extract_relations(self,blocks):
    siblings = {}
    parents = {}
    for i in range(len(blocks)):
      for j in range(i,-1,-1):
        if blocks[i][2] > blocks[j][2]:
          if blocks[i][0]>= blocks[j][0] and blocks[i][1]<= blocks[j][1]:
            parents[str(i)] = j
            break
    for i in range(len(blocks)):
      if (i+1<len(blocks)):
        if blocks[i][2] == blocks[i+1][2]:
            if parents.get(str(i)) == parents.get(str(i+1)):
              siblings[str(i)] = i+1
    return siblings,parents

if __name__ == "__main__":
    code = """
def example_function(param1, param2):
    \"\"\"This is a docstring.\"\"\"
    # This is a comment
    result = param1 + param2
    return result
"""
    analyzer = FunctionAnalyzer()
    print("Function Name:", analyzer.get_function_name(code))
    print("Function Definition Line:", analyzer.get_function_def_line(code))
    print("Docstring:", analyzer.get_docstring(code))
    print("Function Blocks:", analyzer.get_function_blocks(code))
    print("Code Blocks:", analyzer.get_code_blocks(code))
    blocks, block_info = analyzer.get_code_blocks(code)
    siblings, parents = analyzer.extract_relations(block_info)
    print("Siblings:", siblings)
    print("Parents:", parents)
