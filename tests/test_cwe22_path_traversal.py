"""
PoC test for CWE-22: Arbitrary directory deletion via path traversal in /clear_memory endpoint.

The vulnerability flow:
1. POST /init_memory with user_id="../../etc" 
2. Memoryos creates user_data_dir = os.path.join(data_path, "users", "../../etc")
3. POST /clear_memory calls shutil.rmtree(user_data_dir) → deletes arbitrary directory

This test demonstrates the path traversal and verifies that validate_identifier
blocks it. It does NOT require the full Memoryos stack (no OpenAI, no embedding models).
"""
import os
import sys
import re
import tempfile


def test_path_traversal_demonstration():
    """
    Demonstrate that os.path.join with a traversal user_id escapes the data directory.
    This is the core vulnerability — no validation is performed on user_id.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data")
        os.makedirs(data_path, exist_ok=True)
        
        # Simulating what Memoryos.__init__ does (memoryos-playground/memoryos.py line 71)
        malicious_user_id = "../../etc"
        user_data_dir = os.path.join(data_path, "users", malicious_user_id)
        resolved = os.path.realpath(user_data_dir)
        
        # The resolved path escapes the data directory
        data_real = os.path.realpath(data_path)
        assert not resolved.startswith(data_real), \
            f"Expected path to escape data dir, but {resolved} starts with {data_real}"
        print(f"  Traversal confirmed: {user_data_dir} resolves to {resolved}")
        print(f"  This is OUTSIDE {data_real}")


def _load_validate_identifier():
    """Extract validate_identifier from app.py without importing Memoryos."""
    WORKTREE = os.environ.get(
        "WORKTREE",
        os.path.expanduser("~/projects/audits/BAI-LAB-MemoryOS-worktrees/cwe22-app-directory-7b14")
    )
    app_py = os.path.join(WORKTREE, "memoryos-playground", "memdemo", "app.py")
    
    with open(app_py) as f:
        source = f.read()
    
    if "def validate_identifier" not in source:
        raise AssertionError(
            "validate_identifier function not found in app.py — fix not applied yet"
        )
    
    # Extract _SAFE_ID_RE and validate_identifier via exec
    ns = {"re": re, "os": os}
    
    # Extract lines for _SAFE_ID_RE assignment and the function
    lines = source.split('\n')
    code_lines = []
    in_func = False
    func_done = False
    
    for line in lines:
        # Grab the _SAFE_ID_RE compile line
        if '_SAFE_ID_RE' in line and 're.compile' in line:
            code_lines.append(line)
            continue
        
        # Grab the validate_identifier function
        if line.startswith('def validate_identifier'):
            in_func = True
        
        if in_func:
            code_lines.append(line)
            # Detect end of function (next line at indent 0 that isn't blank/comment)
            if func_done and line and not line[0].isspace() and line.strip():
                code_lines.pop()  # remove non-function line
                break
            if len(code_lines) > 1:
                func_done = True
    
    func_source = '\n'.join(code_lines)
    exec(func_source, ns)
    return ns['validate_identifier']


def test_validate_identifier_rejects_traversal():
    """user_id with path traversal sequences must be rejected."""
    validate_identifier = _load_validate_identifier()
    
    malicious_ids = [
        "../../etc",
        "../secret",
        "foo/../../../etc/passwd",
        "..\\windows\\system32",
        "foo/bar",
        "foo\\bar",
        "/absolute/path",
        "user\x00id",
        "",
        "   ",
        "a" * 300,
    ]
    
    for uid in malicious_ids:
        result = validate_identifier(uid)
        assert result is False, f"validate_identifier should reject {uid!r}, but returned True"


def test_validate_identifier_accepts_safe_ids():
    """Normal user_id values must be accepted."""
    validate_identifier = _load_validate_identifier()
    
    safe_ids = [
        "alice",
        "user123",
        "bob_smith",
        "test-user",
        "User.Name",
        "john_doe_42",
    ]
    
    for uid in safe_ids:
        result = validate_identifier(uid)
        assert result is True, f"validate_identifier should accept {uid!r}, but returned False"


if __name__ == "__main__":
    print("Running PoC tests for CWE-22 path traversal in /clear_memory...")
    
    failed = 0
    tests = [
        ("test_path_traversal_demonstration", test_path_traversal_demonstration),
        ("test_validate_identifier_rejects_traversal", test_validate_identifier_rejects_traversal),
        ("test_validate_identifier_accepts_safe_ids", test_validate_identifier_accepts_safe_ids),
    ]
    
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
    
    print(f"\nDone. {len(tests) - failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
