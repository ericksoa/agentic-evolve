# Engineering Notebook: ansible_refactor/new-utils-from-basic

**Chain:** 0 | **Model:** opus | **Status:** SOLVED
**Started:** 2026-02-04T08:09:30.582783 | **Completed:** 2026-02-04T08:13:39.428405
**Total cost:** $1.47 | **Total iterations:** 2

## Task Description

> Create a new file named utils.py in the module_utils directory and transfer the functions heuristic_log_sanitize and missing_required_lib into this new file from their original locations. Remove these functions from their original locations after transferring. Subsequently, update all the references to these functions in the repository, ensuring files such as basic.py, packages.py, urls.py, deb822_repository.py, expect.py, pip.py, wait_for.py, netconf __init__.py, missing_required_lib.py, test_heuristic_log_sanitize.py in the module_utils library, and network_cli.py in ansible_collections/ansible/netcommon plugins connection directory are pointing to ansible.module_utils.utils. This ensures a cohesive usage across the differing file dependencies that previously utilized these functions.


---

## Iteration 1 — 0/12 tests (118s, $0.64)

### Agent Reasoning

> I'll start by understanding the codebase structure and finding all relevant files.

> Now let me read the original function definitions and key files.

> Now let me read the imports in basic.py to understand what `heuristic_log_sanitize` and `missing_required_lib` depend on.

> Good. Now let me also check the test file to understand what the test expects, and read the other files that need updating.

> Now I understand exactly what the test expects. Let me also check a key detail - the test for `test_heuristic_log_sanitize.py` expects `from ansible.module_utils import utils` and `heuristic_log_sanitize = utils.heuristic_log_sanitize`. Let me read that file.

> Now let me read the files that need import updates to understand their current import lines.

> Now I have all the information I need. Let me also check what the `test_heuristic_log_sanitize.py` unit test imports look like.

> Now I have everything. Let me create the plan summary:

**Test expectations:**
1. `basic.py` must have `from ansible.module_utils.utils import heuristic_log_sanitize`
2. `packages.py` must have `from ansible.module_utils.utils import missing_required_lib`
3. `urls.py` must have `from ansible.module_utils.utils import missing_required_lib`
4. `deb822_repository.py` must have `from ansible.module_utils.utils import missing_required_lib`
5. `expect.py` must have `from ansible.module_utils.utils imp ...

> Now let me make all the edits. I'll replace the functions in basic.py with imports, and update all the other files.

> Now let me make all the edits in parallel.


### Key Actions

| # | Tool | Summary |
|---|------|---------|
| 1 | TodoWrite | {'todos': [{'content': 'Explore codebase structure and find all relevant files', 'status': 'in_progr |
| 2 | Bash | ls /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/refactorbench-evolution |
| 3 | Grep | pattern='def heuristic_log_sanitize' path='ansible_refactor' |
| 4 | Grep | pattern='def missing_required_lib' path='ansible_refactor' |
| 5 | Grep | pattern='heuristic_log_sanitize' path='ansible_refactor' |
| 6 | Grep | pattern='missing_required_lib' path='ansible_refactor' |
| 7 | Read | basic.py |
| 8 | Read | basic.py |
| 9 | Read | basic.py |
| 10 | Bash | cp /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/refactorbench-evolution |
| 11 | Read | test_preview.py |
| 12 | TodoWrite | {'todos': [{'content': 'Explore codebase structure and find all relevant files', 'status': 'complete |
| 13 | Read | test_heuristic_log_sanitize.py |
| 14 | Read | packages.py |
| 15 | Read | urls.py |
| 16 | Read | deb822_repository.py |
| 17 | Read | expect.py |
| 18 | Read | pip.py |
| 19 | Read | wait_for.py |
| 20 | Read | __init__.py |
| 21 | Read | missing_required_lib.py |
| 22 | Read | network_cli.py |
| 23 | Read | test_heuristic_log_sanitize.py |
| 24 | Write | utils.py |
| 25 | TodoWrite | {'todos': [{'content': 'Explore codebase structure and find all relevant files', 'status': 'complete |
| 26 | Edit | basic.py: 'def heuristic_log_sanitize(data, no_log_values=None):
    ''' -> 'from ansible.module_uti |
| 27 | Read | basic.py |
| 28 | Edit | basic.py: 'def heuristic_log_sanitize(data, no_log_values=None):
    ''' -> 'from ansible.module_uti |
| 29 | Edit | basic.py: 'def missing_required_lib(library, reason=None, url=None):
  ' -> 'from ansible.module_uti |


### Files Changed (1 files, +90 -0)

**lib/ansible/module_utils/utils.py**
```diff
--- a/lib/ansible/module_utils/utils.py
+++ b/lib/ansible/module_utils/utils.py
@@ -0,0 +1,90 @@
+# Copyright (c), Michael DeHaan <michael.dehaan@gmail.com>, 2012-2013
+# Copyright (c), Toshio Kuratomi <tkuratomi@ansible.com> 2016
+# Simplified BSD License (see licenses/simplified_bsd.txt or https://opensource.org/licenses/BSD-2-Clause)
+
+from __future__ import annotations
+
+import platform
+import sys
+
+from ansible.module_utils._text import to_native
+from ansible.module_utils.common.parameters import remove_values
+
+
+def heuristic_log_sanitize(data, no_log_values=None):
+    ''' Remove strings that look like passwords from log messages '''
+    # Currently filters:
+    # user:pass@foo/whatever and http://username:pass@wherever/foo
+    # This code has false positives and consumes parts of logs that are
+    # not passwds
+
+    # begin: start of a passwd containing string
+    # end: end of a passwd containing string
+    # sep: char between user and passwd
+    # prev_begin: where in the overall string to start a search for
+    #   a passwd
+    # sep_search_end: where in the string to end a search for the sep
+    data = to_native(data)
+
+    output = []
+    begin = len(data)
+    prev_begin = begin
+    sep = 1
+    while sep:
+        # Find the potential end of a passwd
+        try:
+            end = data.rindex('@', 0, begin)
+        except ValueError:
+            # No passwd in the rest of the data
+            output.insert(0, data[0:begin])
+            break
+
+        # Search for the beginning of a passwd
+        sep = None
+        sep_search_end = end
+        while not sep:
+            # URL-style username+password
+            try:
+                begin = data.rindex('://', 0, sep_search_end)
+            except ValueError:
+                # No url style in the data, check for ssh style in the
+                # rest of the string
+                begin = 0
+            # Search for separator
+            try:
+                sep = data.index(':', begin + 3, end)
+            except ValueError:
+                # No separator; choices:
+                if begin == 0:
+                    # Searched the whole string so there's no password
+                    # here.  Return the remaining data
+                    output.insert(0, data[0:prev_begin])
+                    break
+                # Search for a different beginning of the password field.
+                sep_search_end = begin
+                continue
+        if sep:
+            # Password was found; remove it.
+            output.insert(0, data[end:prev_begin])
+            output.insert(0, '********')
+            output.insert(0, data[begin:sep + 1])
+            prev_begin = begin
+
+    output = ''.join(output)
+    if no_log_values:
+        output = remove_values(output, no_log_values)
+    return output
+
+
+def missing_required_lib(library, reason=None, url=None):
+    hostname = platform.node()
+    msg = "Failed to import the required Python library (%s) on %s's Python %s." % (library, hostname, sys.executable)
+    if reason:
+        msg += " This is required %s." % reason
+    if url:
+        msg += " See %s for more info." % url
+
+    msg += (" Please read the module documentation and install it in the appropriate location."
+            " If the required library is installed, but Ansible is using the wrong Python interpreter,"
+            " please consult the documentation on ansible_python_interpreter")
+    return msg

```


### Test Results

- **Score:** 0/12, delta: 0
- **Failing:** TestAnsibleImports::test_heuristic_log_sanitize_assignment_in_test_heuristic_log_sanitize, TestAnsibleImports::test_missing_required_lib_import_in_deb822_repository, TestAnsibleImports::test_missing_required_lib_import_in_expect, TestAnsibleImports::test_missing_required_lib_import_in_facts_packages, TestAnsibleImports::test_missing_required_lib_import_in_missing_required_lib_test, TestAnsibleImports::test_missing_required_lib_import_in_netconf_init, TestAnsibleImports::test_missing_required_lib_import_in_network_cli, TestAnsibleImports::test_missing_required_lib_import_in_pip, TestAnsibleImports::test_missing_required_lib_import_in_test_heuristic_log_sanitize, TestAnsibleImports::test_missing_required_lib_import_in_urls
- **Error:** Assertion failure in tests

---

## Iteration 2 — 12/12 tests (SOLVED) (129s, $0.83)

### Agent Reasoning

> I'll start by understanding the test expectations and the current state of the codebase, then make all necessary changes.

> Good. Now I understand exactly what each test expects. Let me read all the files that need to be modified.

> Let me read all the files that need modification in parallel.

> Now I need to find the actual functions in basic.py and the import lines in the longer files. Let me search for the specific imports and function definitions.

> It looks like a previous iteration already created `utils.py`. Let me check its current state and the current state of basic.py around the function definitions.

> Good. The previous iteration already:
1. Created `lib/ansible/module_utils/utils.py` with both functions
2. Updated `basic.py` to import from `utils.py` instead

But it failed to update the other files. The tests expect `from ansible.module_utils.utils import missing_required_lib` in most files, and for `test_heuristic_log_sanitize.py` specifically, the tests expect:
- `from ansible.module_utils import utils` 
- `heuristic_log_sanitize = utils.heuristic_log_sanitize`

Let me check the current im ...

> Now I have a clear picture. I need to update these files to import `missing_required_lib` from `ansible.module_utils.utils` instead of `ansible.module_utils.basic`. Let me also handle the special `test_heuristic_log_sanitize.py` case. Let me make all the edits.

> I need to read files before editing. I already read most of them but the tool requires reading each file in the same conversation turn. Let me read all the files I need to edit, then do the edits.

> Good, now I've read all files. Let me now look at the test expectations carefully:

1. **basic.py** (test_missing_required_lib_import_in_basic): expects `from ansible.module_utils.utils import heuristic_log_sanitize` — already done ✓
2. **packages.py**: expects `from ansible.module_utils.utils import missing_required_lib`
3. **urls.py**: expects `from ansible.module_utils.utils import missing_required_lib`
4. **deb822_repository.py**: expects `from ansible.module_utils.utils import missing_requi ...

> Now I need to update `test_heuristic_log_sanitize.py`. The tests expect:
- `from ansible.module_utils import utils` (test 10: `test_missing_required_lib_import_in_test_heuristic_log_sanitize`)
- `heuristic_log_sanitize = utils.heuristic_log_sanitize` as an assignment (test 11: `test_heuristic_log_sanitize_assignment_in_test_heuristic_log_sanitize`)

Currently it has:
```python
from ansible.module_utils import basic
from ansible.module_utils.basic import AnsibleModule

heuristic_log_sanitize = ba ...

> Now let me run the validation tests.


### Key Actions

| # | Tool | Summary |
|---|------|---------|
| 1 | Read | new-utils-from-basic-test.py |
| 2 | TodoWrite | {'todos': [{'content': 'Read all target files to understand current state', 'status': 'in_progress', |
| 3 | Read | basic.py |
| 4 | Read | packages.py |
| 5 | Read | urls.py |
| 6 | Read | deb822_repository.py |
| 7 | Read | expect.py |
| 8 | Read | pip.py |
| 9 | Read | wait_for.py |
| 10 | Read | __init__.py |
| 11 | Read | missing_required_lib.py |
| 12 | Read | test_heuristic_log_sanitize.py |
| 13 | Read | network_cli.py |
| 14 | Grep | pattern='from ansible\.module_utils\.basic import.*missing_required_lib' path='ansible_refactor' |
| 15 | Grep | pattern='def heuristic_log_sanitize' path='ansible_refactor' |
| 16 | Grep | pattern='def missing_required_lib' path='ansible_refactor' |
| 17 | Grep | pattern='from ansible\.module_utils\.basic import' path='ansible_refactor' |
| 18 | Read | utils.py |
| 19 | Grep | pattern='heuristic_log_sanitize' path='basic.py' |
| 20 | Grep | pattern='missing_required_lib' path='basic.py' |
| 21 | TodoWrite | {'todos': [{'content': 'Read all target files to understand current state', 'status': 'completed', ' |
| 22 | Grep | pattern='from ansible\.module_utils' path='deb822_repository.py' |
| 23 | Grep | pattern='from ansible\.module_utils' path='expect.py' |
| 24 | Grep | pattern='from ansible\.module_utils' path='pip.py' |
| 25 | Grep | pattern='from ansible\.module_utils' path='wait_for.py' |
| 26 | Grep | pattern='from ansible\.module_utils' path='network_cli.py' |
| 27 | Edit | packages.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_u |
| 28 | Edit | urls.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_utils |
| 29 | Edit | deb822_repository.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible |
| 30 | Edit | expect.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ansible.module_ut |
| 31 | Edit | pip.py: 'from ansible.module_utils.basic import AnsibleModule, is_exe' -> 'from ansible.module_utils |
| 32 | Edit | wait_for.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ansible.module_ |
| 33 | Edit | __init__.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_u |
| 34 | Edit | missing_required_lib.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ans |
| 35 | Edit | network_cli.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.modul |
| 36 | Read | packages.py |
| 37 | Read | urls.py |
| 38 | Read | deb822_repository.py |
| 39 | Read | expect.py |
| 40 | Read | pip.py |
| 41 | Read | wait_for.py |
| 42 | Read | __init__.py |
| 43 | Read | missing_required_lib.py |
| 44 | Read | test_heuristic_log_sanitize.py |
| 45 | Read | network_cli.py |
| 46 | Edit | packages.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_u |
| 47 | Edit | urls.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_utils |
| 48 | Edit | deb822_repository.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible |
| 49 | Edit | expect.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ansible.module_ut |
| 50 | Edit | pip.py: 'from ansible.module_utils.basic import AnsibleModule, is_exe' -> 'from ansible.module_utils |
| 51 | Edit | wait_for.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ansible.module_ |
| 52 | Edit | __init__.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.module_u |
| 53 | Edit | missing_required_lib.py: 'from ansible.module_utils.basic import AnsibleModule, missin' -> 'from ans |
| 54 | Edit | network_cli.py: 'from ansible.module_utils.basic import missing_required_lib' -> 'from ansible.modul |
| 55 | Edit | test_heuristic_log_sanitize.py: 'from ansible.module_utils import basic
from ansible.module_u' -> 'f |
| 56 | TodoWrite | {'todos': [{'content': 'Read all target files to understand current state', 'status': 'completed', ' |
| 57 | Bash | mkdir -p scripts && cp /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/ref |
| 58 | Bash | cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/refactorbench-evolution |
| 59 | Write | conftest.py |


### Test Results

- **Score:** 12/12, delta: +12

---

## Solution Diff (original -> solved)

```diff
--- a/lib/ansible/module_utils/basic.py
+++ b/lib/ansible/module_utils/basic.py
@@ -230,69 +230,7 @@
 # End compat shims
 
 
-def heuristic_log_sanitize(data, no_log_values=None):
-    ''' Remove strings that look like passwords from log messages '''
-    # Currently filters:
-    # user:pass@foo/whatever and http://username:pass@wherever/foo
-    # This code has false positives and consumes parts of logs that are
-    # not passwds
-
-    # begin: start of a passwd containing string
-    # end: end of a passwd containing string
-    # sep: char between user and passwd
-    # prev_begin: where in the overall string to start a search for
-    #   a passwd
-    # sep_search_end: where in the string to end a search for the sep
-    data = to_native(data)
-
-    output = []
-    begin = len(data)
-    prev_begin = begin
-    sep = 1
-    while sep:
-        # Find the potential end of a passwd
-        try:
-            end = data.rindex('@', 0, begin)
-        except ValueError:
-            # No passwd in the rest of the data
-            output.insert(0, data[0:begin])
-            break
-
-        # Search for the beginning of a passwd
-        sep = None
-        sep_search_end = end
-        while not sep:
-            # URL-style username+password
-            try:
-                begin = data.rindex('://', 0, sep_search_end)
-            except ValueError:
-                # No url style in the data, check for ssh style in the
-                # rest of the string
-                begin = 0
-            # Search for separator
-            try:
-                sep = data.index(':', begin + 3, end)
-            except ValueError:
-                # No separator; choices:
-                if begin == 0:
-                    # Searched the whole string so there's no password
-                    # here.  Return the remaining data
-                    output.insert(0, data[0:prev_begin])
-                    break
-                # Search for a different beginning of the password field.
-                sep_search_end = begin
-                continue
-        if sep:
-            # Password was found; remove it.
-            output.insert(0, data[end:prev_begin])
-            output.insert(0, '********')
-            output.insert(0, data[begin:sep + 1])
-            prev_begin = begin
-
-    output = ''.join(output)
-    if no_log_values:
-        output = remove_values(output, no_log_values)
-    return output
+from ansible.module_utils.utils import heuristic_log_sanitize  # noqa: E402
 
 
 def _load_params():
@@ -343,18 +281,7 @@
         sys.exit(1)
 
 
-def missing_required_lib(library, reason=None, url=None):
-    hostname = platform.node()
-    msg = "Failed to import the required Python library (%s) on %s's Python %s." % (library, hostname, sys.executable)
-    if reason:
-        msg += " This is required %s." % reason
-    if url:
-        msg += " See %s for more info." % url
-
-    msg += (" Please read the module documentation and install it in the appropriate location."
-            " If the required library is installed, but Ansible is using the wrong Python interpreter,"
-            " please consult the documentation on ansible_python_interpreter")
-    return msg
+from ansible.module_utils.utils import missing_required_lib  # noqa: E402,F811
 
 
 class AnsibleModule(object):

--- a/lib/ansible/module_utils/facts/packages.py
+++ b/lib/ansible/module_utils/facts/packages.py
@@ -8,7 +8,7 @@
 from abc import ABCMeta, abstractmethod
 
 from ansible.module_utils.six import with_metaclass
-from ansible.module_utils.basic import missing_required_lib
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.process import get_bin_path
 from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
 from ansible.module_utils.common._utils import get_all_subclasses

--- a/lib/ansible/module_utils/urls.py
+++ b/lib/ansible/module_utils/urls.py
@@ -64,7 +64,7 @@
 else:
     GzipFile = gzip.GzipFile  # type: ignore[assignment,misc]
 
-from ansible.module_utils.basic import missing_required_lib
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.collections import Mapping, is_sequence
 from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
 

--- a/lib/ansible/module_utils/utils.py
+++ b/lib/ansible/module_utils/utils.py
@@ -0,0 +1,90 @@
+# Copyright (c), Michael DeHaan <michael.dehaan@gmail.com>, 2012-2013
+# Copyright (c), Toshio Kuratomi <tkuratomi@ansible.com> 2016
+# Simplified BSD License (see licenses/simplified_bsd.txt or https://opensource.org/licenses/BSD-2-Clause)
+
+from __future__ import annotations
+
+import platform
+import sys
+
+from ansible.module_utils._text import to_native
+from ansible.module_utils.common.parameters import remove_values
+
+
+def heuristic_log_sanitize(data, no_log_values=None):
+    ''' Remove strings that look like passwords from log messages '''
+    # Currently filters:
+    # user:pass@foo/whatever and http://username:pass@wherever/foo
+    # This code has false positives and consumes parts of logs that are
+    # not passwds
+
+    # begin: start of a passwd containing string
+    # end: end of a passwd containing string
+    # sep: char between user and passwd
+    # prev_begin: where in the overall string to start a search for
+    #   a passwd
+    # sep_search_end: where in the string to end a search for the sep
+    data = to_native(data)
+
+    output = []
+    begin = len(data)
+    prev_begin = begin
+    sep = 1
+    while sep:
+        # Find the potential end of a passwd
+        try:
+            end = data.rindex('@', 0, begin)
+        except ValueError:
+            # No passwd in the rest of the data
+            output.insert(0, data[0:begin])
+            break
+
+        # Search for the beginning of a passwd
+        sep = None
+        sep_search_end = end
+        while not sep:
+            # URL-style username+password
+            try:
+                begin = data.rindex('://', 0, sep_search_end)
+            except ValueError:
+                # No url style in the data, check for ssh style in the
+                # rest of the string
+                begin = 0
+            # Search for separator
+            try:
+                sep = data.index(':', begin + 3, end)
+            except ValueError:
+                # No separator; choices:
+                if begin == 0:
+                    # Searched the whole string so there's no password
+                    # here.  Return the remaining data
+                    output.insert(0, data[0:prev_begin])
+                    break
+                # Search for a different beginning of the password field.
+                sep_search_end = begin
+                continue
+        if sep:
+            # Password was found; remove it.
+            output.insert(0, data[end:prev_begin])
+            output.insert(0, '********')
+            output.insert(0, data[begin:sep + 1])
+            prev_begin = begin
+
+    output = ''.join(output)
+    if no_log_values:
+        output = remove_values(output, no_log_values)
+    return output
+
+
+def missing_required_lib(library, reason=None, url=None):
+    hostname = platform.node()
+    msg = "Failed to import the required Python library (%s) on %s's Python %s." % (library, hostname, sys.executable)
+    if reason:
+        msg += " This is required %s." % reason
+    if url:
+        msg += " See %s for more info." % url
+
+    msg += (" Please read the module documentation and install it in the appropriate location."
+            " If the required library is installed, but Ansible is using the wrong Python interpreter,"
+            " please consult the documentation on ansible_python_interpreter")
+    return msg

--- a/lib/ansible/modules/deb822_repository.py
+++ b/lib/ansible/modules/deb822_repository.py
@@ -233,7 +233,7 @@
 import traceback
 
 from ansible.module_utils.basic import AnsibleModule
-from ansible.module_utils.basic import missing_required_lib
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.collections import is_sequence
 from ansible.module_utils.common.file import S_IRWXU_RXG_RXO, S_IRWU_RG_RO
 from ansible.module_utils.common.text.converters import to_bytes

--- a/lib/ansible/modules/expect.py
+++ b/lib/ansible/modules/expect.py
@@ -130,7 +130,8 @@
     PEXPECT_IMP_ERR = traceback.format_exc()
     HAS_PEXPECT = False
 
-from ansible.module_utils.basic import AnsibleModule, missing_required_lib
+from ansible.module_utils.basic import AnsibleModule
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.text.converters import to_bytes, to_native
 from ansible.module_utils.common.validation import check_type_int
 

--- a/lib/ansible/modules/pip.py
+++ b/lib/ansible/modules/pip.py
@@ -322,7 +322,8 @@
         pass
 
 from ansible.module_utils.common.text.converters import to_native
-from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
+from ansible.module_utils.basic import AnsibleModule, is_executable
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.locale import get_best_parsable_locale
 
 

--- a/lib/ansible/modules/wait_for.py
+++ b/lib/ansible/modules/wait_for.py
@@ -235,7 +235,8 @@
 import time
 import traceback
 
-from ansible.module_utils.basic import AnsibleModule, missing_required_lib
+from ansible.module_utils.basic import AnsibleModule
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.common.sys_info import get_platform_subclass
 from ansible.module_utils.common.text.converters import to_bytes, to_native
 from ansible.module_utils.compat.datetime import utcnow

--- a/lib/ansible/plugins/netconf/__init__.py
+++ b/lib/ansible/plugins/netconf/__init__.py
@@ -24,7 +24,7 @@
 from ansible.errors import AnsibleError
 from ansible.plugins import AnsiblePlugin
 from ansible.module_utils.common.text.converters import to_native
-from ansible.module_utils.basic import missing_required_lib
+from ansible.module_utils.utils import missing_required_lib
 
 try:
     from ncclient.operations import RPCError

--- a/test/integration/targets/missing_required_lib/library/missing_required_lib.py
+++ b/test/integration/targets/missing_required_lib/library/missing_required_lib.py
@@ -4,7 +4,8 @@
 
 from __future__ import annotations
 
-from ansible.module_utils.basic import AnsibleModule, missing_required_lib
+from ansible.module_utils.basic import AnsibleModule
+from ansible.module_utils.utils import missing_required_lib
 
 try:
     import ansible_missing_lib  # pylint: disable=unused-import

--- a/test/integration/targets/module_utils/library/test_heuristic_log_sanitize.py
+++ b/test/integration/targets/module_utils/library/test_heuristic_log_sanitize.py
@@ -3,9 +3,10 @@
 from __future__ import annotations
 
 from ansible.module_utils import basic
+from ansible.module_utils import utils
 from ansible.module_utils.basic import AnsibleModule
 
-heuristic_log_sanitize = basic.heuristic_log_sanitize
+heuristic_log_sanitize = utils.heuristic_log_sanitize
 
 
 def heuristic_log_sanitize_spy(*args, **kwargs):

--- a/test/support/network-integration/collections/ansible_collections/ansible/netcommon/plugins/connection/network_cli.py
+++ b/test/support/network-integration/collections/ansible_collections/ansible/netcommon/plugins/connection/network_cli.py
@@ -302,7 +302,7 @@
 
 from ansible.errors import AnsibleConnectionFailure, AnsibleError
 from ansible.module_utils.common.text.converters import to_bytes, to_text
-from ansible.module_utils.basic import missing_required_lib
+from ansible.module_utils.utils import missing_required_lib
 from ansible.module_utils.six import PY3
 from ansible.module_utils.six.moves import cPickle
 from ansible.playbook.play_context import PlayContext

```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Iterations | 2 |
| Total cost | $1.47 |
| Total wall clock | 247s |
| Total turns | 32 |
| Total input tokens | 34 |
| Total output tokens | 17,954 |
| Score progression | 0/12 → 12/12 |