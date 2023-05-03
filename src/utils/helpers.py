# import libraries
import json
import subprocess
import re
import os
import sys
import traceback
import chardet
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

# get user path
user_path = os.path.expanduser("~")
pandoc_path = os.path.join(
    user_path, "AppData", "Local", "Pandoc", "pandoc.exe")


# ----- Github -----#

# function to get github docs from a repo url and convert to md
def get_github_docs(repo_url: str):
    '''Get documents from a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Yields:
        '''

    print(f"Processing repository: {repo_url}")
    repo_owner, repo_name, subdirectory = extract_github_info(repo_url)
    print(f"Subdirectory: {subdirectory}")

    with tempfile.TemporaryDirectory() as d:
        repo_path = pathlib.Path(d)

        # Initialize an empty Git repository
        init_cmd = "git init"
        subprocess.check_call(init_cmd, cwd=d, shell=True)

        # Add the remote repository
        remote_cmd = f"git remote add origin https://github.com/{repo_owner}/{repo_name}.git"
        subprocess.check_call(remote_cmd, cwd=d, shell=True)

        # Enable sparse-checkout
        sparse_cmd = "git config core.sparseCheckout true"
        subprocess.check_call(sparse_cmd, cwd=d, shell=True)

        # Specify the subdirectory to include
        if subdirectory is not None:
            if subdirectory.startswith("tree/"):
                _, branch, *subdir_parts = subdirectory.split("/")
                subdirectory = "/".join(subdir_parts)
            else:
                branch = "HEAD"

            with open(repo_path / ".git/info/sparse-checkout", "w") as f:
                f.write(subdirectory)

                # Fetch the specified branch
                fetch_cmd = f"git fetch --depth 1 origin {branch}"
                subprocess.check_call(fetch_cmd, cwd=d, shell=True)
        else:
            # Fetch the default branch
            fetch_cmd = "git fetch --depth 1 origin"
            subprocess.check_call(fetch_cmd, cwd=d, shell=True)

        # Checkout the fetched branch
        checkout_cmd = "git checkout FETCH_HEAD"
        subprocess.check_call(checkout_cmd, cwd=d, shell=True)

        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)

        matched_files = list(repo_path.glob("**/*.*"))
        print(f"Matched files: {matched_files}")
        venv_files = 0
        other_files = 0

        for file in matched_files:
            if ".venv" in str(file.parts):
                venv_files += 1
                continue
            try:
                relative_path = file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                file_ext = file.suffix.lower()

                # Detect encoding
                with open(file, "rb") as f:
                    content = f.read()
                result = charset_normalizer.detect(content)
                encoding = result["encoding"]
                print(f'Encoding: {encoding}')

                if file_ext == ".rst":
                    page_content = helpers.convert_rst_to_md(
                        str(file), encoding)
                elif file_ext == ".ipynb":
                    page_content = helpers.convert_ipynb_to_md(
                        str(file), encoding)
                elif file_ext == ".py":
                    page_content = helpers.convert_py_to_md(
                        str(file), encoding)
                else:
                    page_content = helpers.convert_any_to_md(
                        str(file), encoding)

                if page_content is not None:
                    yield Document(page_content=page_content, metadata={"source": github_url})

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                other_files += 1
                continue

        print(f"Total venv files skipped: {venv_files}")
        print(f"Total other files skipped: {other_files} (CHECK THESE FILES!)")

# def extract_github_info(repo_url: str):
#     parsed_url = urlparse(repo_url)
#     print(f'Parsed URL: {parsed_url}')
#     path_parts = parsed_url.path.strip("/").split("/")
#     print(f'Path parts: {path_parts}')
#     repo_owner = path_parts[0]
#     repo_name = path_parts[1]
#     subdirectory = "/".join(path_parts[4:]) if len(path_parts) > 4 else None
#     return repo_owner, repo_name, subdirectory


def extract_github_info(repo_url: str):
    print(f'DEBUG Repo URL: {repo_url}')
    parsed_url = urlparse(repo_url)
    print(f'DEBUG Parsed URL: {parsed_url}')
    path_parts = parsed_url.path.strip("/").split("/")
    repo_owner = path_parts[0]
    repo_name = path_parts[1]
    subdirectory = None
    if len(path_parts) > 3 and path_parts[2] == "tree":
        subdirectory = "tree/" + "/".join(path_parts[3:])
    return repo_owner, repo_name, subdirectory


# ----- Convert and output -----#

# any to md
def convert_any_to_md(filepath, encoding):

    with open(filepath, "r", encoding=encoding) as file:
        encoded_text = file.read()

    md_content = f"\n{encoded_text}\n"
    print(f'Successfully converted {filepath} to markdown')
    return md_content

# ipynb to md


def convert_ipynb_to_md(filepath, encoding):
    try:
        print(f"Trying to convert {filepath}")
        update_nbformat_minor(filepath)

        try:
            # Try opening the file with the encoding detected by charset-normalizer
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                nb = nbformat.read(f, as_version=4)
        except UnicodeDecodeError:
            # Fallback to chardet if charset-normalizer fails
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
            detected_encoding = result['encoding']

            with open(filepath, 'r', encoding=detected_encoding, errors='replace') as f:
                nb = nbformat.read(f, as_version=4)

        # Check and fix metadata
        if "metadata" not in nb:
            nb.metadata = {}
        if "language" not in nb.metadata:
            nb.metadata["language"] = "python"

        md_exporter = MarkdownExporter()
        (body, resources) = md_exporter.from_notebook_node(nb)

        print(f'Successfully converted {filepath} to markdown')
        return body

    except Exception as e:
        print(f"Error occurred while converting {filepath} to markdown")
        print(f"Error: {e}")
        return None

    except Exception as e:
        print(f"Error occurred while converting {filepath} to markdown")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_tb(e.__traceback__)
        sys.exit(1)


# rst to md
def convert_rst_to_md(filepath, encoding):
    output_file = 'temp_output.md'
    command = [pandoc_path, filepath, '-f', 'rst',
               '-t', 'markdown', '-o', output_file]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f'Successfully converted {filepath} to markdown')

        # Read the converted file and remove custom blocks
        with open(output_file, "r", encoding=encoding) as f:
            md_content = f.read()

        md_content = re.sub(r':::.*?:::', '', md_content, flags=re.DOTALL)

        # Remove the temporary output file
        os.remove(output_file)

        return md_content
    else:
        print(f'Error during conversion: {result.stderr}')
        return None

# py to md


def convert_py_to_md(filepath, encoding):
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


def update_nbformat_minor(filepath):
    with open(filepath, 'r') as file:
        notebook = json.load(file)

    # Update the nbformat_minor value
    if notebook.get('nbformat_minor') is not None:
        notebook['nbformat_minor'] = 5
    else:
        print("nbformat_minor not found in the file")

    with open(filepath, 'w') as file:
        json.dump(notebook, file, indent=2)
