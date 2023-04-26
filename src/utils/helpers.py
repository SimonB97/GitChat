# import libraries
import subprocess
import re
import os
import nbformat
from nbconvert import MarkdownExporter
import charset_normalizer

from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage
import os
import pathlib
import subprocess
import tempfile
from urllib.parse import urlparse
import utils.helpers as helpers

pandoc_path = "C:/Users/sbene/appdata/local/pandoc/pandoc.exe"


#----- Github -----#

# function to get github docs from a repo url and convert to md
def get_github_docs(repo_url: str):
    '''Get documents from a GitHub repository.
    
    Args:
        repo_url (str): The URL of the GitHub repository.
        
    Yields:
        '''
    repo_owner, repo_name, subdirectory = extract_github_info(repo_url)
    
    with tempfile.TemporaryDirectory() as d:
        clone_cmd = f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git ."
        
        if subdirectory:
            clone_cmd += f" --filter=blob:none --no-checkout --sparse"
        
        subprocess.check_call(clone_cmd, cwd=d, shell=True)
        
        if subdirectory:
            subprocess.check_call(f"git sparse-checkout init --cone", cwd=d, shell=True)
            subprocess.check_call(f"git sparse-checkout set {subdirectory}", cwd=d, shell=True)
            subprocess.check_call(f"git checkout", cwd=d, shell=True)
        
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        file_patterns = ["*.md", "*.mdx", "*.rst", "*.ipynb", "*.py", "*.yaml", "*.html"]
        matched_files = []
        
        for pattern in file_patterns:
            matched_files.extend(list(repo_path.glob(f"**/{pattern}")))
        
        for file in matched_files:
            with open(file, "r", encoding="utf-8") as f:
                relative_path = file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                file_ext = file.suffix.lower()
                if file_ext in [".md", ".mdx"]:
                    page_content = f.read()
                elif file_ext == ".rst":
                    page_content = helpers.convert_rst_to_md(str(file))
                elif file_ext == ".ipynb":
                    page_content = helpers.convert_ipynb_to_md(str(file))
                elif file_ext == ".py":
                    page_content = helpers.convert_py_to_md(str(file))
                # yaml
                elif file_ext in [".yaml", ".html"]:
                    page_content = f.read()
                else:
                    continue

                if page_content is not None:
                    yield Document(page_content=page_content, metadata={"source": github_url})

def extract_github_info(repo_url: str):
    parsed_url = urlparse(repo_url)
    print(f'Parsed URL: {parsed_url}')
    path_parts = parsed_url.path.strip("/").split("/")
    print(f'Path parts: {path_parts}')
    repo_owner = path_parts[0]
    repo_name = path_parts[1]
    subdirectory = "/".join(path_parts[4:]) if len(path_parts) > 4 else None
    return repo_owner, repo_name, subdirectory


#----- Convert and output -----#

# ipynb to md
def convert_ipynb_to_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        md_exporter = MarkdownExporter()
        (body, resources) = md_exporter.from_notebook_node(nb)

    print(f'Successfully converted {filepath} to markdown')
    return body

# rst to md
def convert_rst_to_md(filepath):
    output_file = 'temp_output.md'
    command = [pandoc_path, filepath, '-f', 'rst', '-t', 'markdown', '-o', output_file]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f'Successfully converted {filepath} to markdown')
        
        # Read the converted file and remove custom blocks
        with open(output_file, "r", encoding='utf-8') as f:
            md_content = f.read()

        md_content = re.sub(r':::.*?:::', '', md_content, flags=re.DOTALL)

        # Remove the temporary output file
        os.remove(output_file)

        return md_content
    else:
        print(f'Error during conversion: {result.stderr}')
        return None
    
# py to md
def convert_py_to_md(filepath):
    with open(filepath, "rb") as file:
        content = file.read()

    result = charset_normalizer.detect(content)
    encoding = result["encoding"]
    print(f'Encoding: {encoding}')

    with open(filepath, "r", encoding=encoding) as file:
        code = file.read()
    
    md_content = f"python\n{code}\n"
    print(f'Successfully converted {filepath} to markdown')
    return md_content  

# compile messages (chat history)
def compile_messages(messages):
    compiled_string = ""

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "USER Input was:"
        elif isinstance(message, AIMessage):
            role = "AI message was:"
        else:
            continue

        compiled_string += f"{role}\n\"{message.content}\"\n\n"

    return compiled_string

