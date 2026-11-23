# DocDrift: second annotation pass (annotator B)

For each pair, read the signature and the docstring diff (`-` = old, `+` = new) and write ONE label
after `LABEL:`. Do not look at annotator A's file first (exp8b_annotation_A.csv).

- **factual**: the old docstring was *wrong* about the code (wrong name, wrong type, wrong behaviour, outdated claim)
- **incomplete**: the old docstring *omitted* something the code does (undocumented param, return, exception, condition)
- **clarity**: rewording / restructuring / examples; the old docstring was not wrong or missing anything important
- **noise**: typo, formatting, markup, links, see-also, version notes, or an edit that made the docstring *worse*

Look up the commit on GitHub (`https://github.com/<repo>/commit/<commit>`) if the diff is not enough.

## 1. pair 6730 — pandas-dev/pandas `d450387fa6`
*DOC: Add notes explaining params is db driver dep. GH7573*

```
def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None):
```
```diff
 params : list, tuple or dict, optional
-    List of parameters to pass to execute method.
+    List of parameters to pass to execute method.  The syntax used
+    to pass parameters is database driver dependent. Check your 
+    database driver documentation for which of the five syntax styles,
+    described in PEP 249's paramstyle, is supported.
+    Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
 parse_dates : list or dict
```

LABEL: 

## 2. pair 1433 — sphinx-doc/sphinx `01ffa6cf04`
*updating docstring for add_js_file for body element*

```
def add_js_file(self, filename: str, **kwargs: str) -> None:
```
```diff
 template will include.  The filename must be relative to the HTML
-static path , or a full URI with scheme.  The keyword arguments are
-also accepted for attributes of ``<script>`` tag.
+static path , or a full URI with scheme.  If the keyword argument
+``body`` is given, its value will be added between the
+``<script>`` tags. Extra keyword arguments are included as
+attributes of the ``<script>`` tag.
 
     # => <script src="_static/example.js" async="async"></script>
+    
+    app.add_js_file(None, body="var myVariable = 'foo';")
+    # => <script>var myVariable = 'foo';</script>
 
```

LABEL: 

## 3. pair 3098 — numpy/numpy `923e16db07`
*Documentation and misc. typos*

```
def _ureduce(a, func, **kwargs):
```
```diff
     Reduction function capable of receiving a single axis argument.
-    It is is called with `a` as first argument followed by `kwargs`.
+    It is called with `a` as first argument followed by `kwargs`.
 kwargs : keyword arguments
```

LABEL: 

## 4. pair 2768 — numpy/numpy `2ef7dc370e`
*DOC: add `linalg.cross` example*

```
def cross(x1, x2, /, *, axis=-1):
```
```diff
 numpy.cross
+
+Examples
+--------
+Vector cross-product.
+
+>>> x = np.array([1, 2, 3])
+>>> y = np.array([4, 5, 6])
+>>> np.linalg.cross(x, y)
+array([-3,  6, -3])
+
+Multiple vector cross-products. Note that the direction of the cross
+product vector is defined by the *right-hand rule*.
+
+>>> x = np.array([[1,2,3], [4,5,6]])
+>>> y = np.array([[4,5,6], [1,2,3]])
+>>> np.cross(x, y)
+array([[-3,  6, -3],
+       [ 3, -6,  3]])
+
+>>> x = np.array([[1, 2], [3, 4], [5, 6]])
+>>> y = np.array([[4, 5], [6, 1], [2, 3]])
+>>> np.cross(x, y, axis=0)
+array([[-24,  6],
+       [ 18, 24],
+       [-6,  -18]])
```

LABEL: 

## 5. pair 6837 — pypa/setuptools `d05bcc2e38`
*Avoid path separator problem in doctest on windows*

```
def _expand_patterns(cls, patterns: list[str]) -> Iterator[str]:
```
```diff
 ['pyproject.toml', 'LICENSE']
+>>> list(Distribution._expand_patterns(['setuptools/**/pyprojecttoml.py']))
+['setuptools/config/pyprojecttoml.py']
+>>> list(Distribution._expand_patterns(['../LICENSE']))
+Traceback (most recent call last):
+...
+setuptools.errors.InvalidConfigError: Pattern '../LICENSE' cannot contain '..'
```

LABEL: 

## 6. pair 14 — pallets/flask `25ab05e6a2`
*remove old note about InternalServerError*

```
def handle_exception(self, e):
```
```diff
 
-.. note::
-    Prior to Werkzeug 1.0.0, ``InternalServerError`` will not
-    always have an ``original_exception`` attribute. Use
-    ``getattr(e, "original_exception", None)`` to simulate the
-    behavior for compatibility.
-
 .. versionchanged:: 1.1.0
```

LABEL: 

## 7. pair 938 — tornadoweb/tornado `b79f82aa62`
*Allow TCPServer.handle_stream to be a coroutine*

```
def handle_stream(self, stream, address):
```
```diff
 Override to handle a new `.IOStream` from an incoming connection.
+
+This method may be a coroutine; if so any exceptions it raises
+asynchronously will be logged. Accepting of incoming connections
+will not be blocked by this coroutine.
+
+If this `TCPServer` is configured for SSL, ``handle_stream``
+may be called before the SSL handshake has completed. Use
+`.SSLIOStream.wait_for_handshake` if you need to verify the client's
+certificate or use NPN/ALPN.
+
+.. versionchanged:: 4.2
+   Added the option for this method to be a coroutine.
```

LABEL: 

## 8. pair 1180 — aio-libs/aiohttp `7054639004`
*Fix docstrings for req.host and req.remote (#2607)*

```
def remote(self):
```
```diff
 
-The IP is resolved through the following headers, in this order:
+The IP is resolved in this order:
 
-- Forwarded
-- X-Forwarded-For
+- overridden value by .clone(remote=new_remote) call.
 - peername of opened socket
```

LABEL: 

## 9. pair 6827 — pypa/setuptools `91d8cdc9da`
*Address review feedback.*

```
def get_export_symbols(self, ext: Extension) -> list[str]:
```
```diff
 Return the list of symbols that a shared extension has to
-export.  This returns, and possibly updates, 'ext.export_symbols'.
-
-On Python 3.14 and below (that is, before a new export hook name was added),
-it adds "PyInit_" + module_name to 'ext.export_symbols'.
-Only relevant on Windows, where the .pyd file (DLL) must export the module
-import hook function.
-
-Since Python 3.15, nothing is added.
-An export directive was added to the "PyMODINIT_FUNC" implementation
-for Python 3.15, which is all that is needed.
+export.  This returns, and possibly extends, 'ext.export_symbols'
+with any symbols implied for the running Python (see
+'implied_symbols').  Only relevant on Windows, where the .pyd file
+(DLL) must export the module import hook function.
```

LABEL: 

## 10. pair 5748 — matplotlib/matplotlib `1c200302cf`
*Fix docstring type*

```
def set_prop_cycle(self, *args, **kwargs):
```
```diff
 
-label : name
+label : str
     The property key. Must be a valid `Artist` property.
```

LABEL: 

## 11. pair 6860 — pypa/setuptools `43a44e3cba`
*Add tests capturing failure. Ref pypa/distutils#164.*

```
def find_library_file(self, dirs, lib, debug=0):
```
```diff
 ignoring even GCC's "-static" option.
+
+>>> compiler = UnixCCompiler()
+>>> compiler._library_root = lambda dir: dir
+>>> monkeypatch = getfixture('monkeypatch')
+>>> monkeypatch.setattr(os.path, 'exists', lambda d: 'existing' in d)
+>>> dirs = ('/foo/bar/missing', '/foo/bar/existing')
+>>> compiler.find_library_file(dirs, 'abc')
+'/foo/bar/existing/libabc.dylib'
+>>> compiler.find_library_file(reversed(dirs), 'abc')
+'/foo/bar/existing/libabc.dylib'
+>>> monkeypatch.setattr(os.path, 'exists',
+...     lambda d: 'existing' in d and '.a' in d)
+>>> compiler.find_library_file(dirs, 'abc')
+'/foo/bar/existing/libabc.a'
+>>> compiler.find_library_file(reversed(dirs), 'abc')
+'/foo/bar/existing/libabc.a'
```

LABEL: 

## 12. pair 1422 — sphinx-doc/sphinx `42e491a4c2`
*refactor: AttributeError handling for getslots() is not needed*

```
def getslots(obj: Any) -> Optional[Dict]:
```
```diff
 Return None if gienv *obj* does not have __slots__.
-Raises AttributeError if given *obj* raises an error on accessing __slots__.
 Raises TypeError if given *obj* is not a class.
```

LABEL: 

## 13. pair 456 — python-jsonschema/jsonschema `fcb178774f`
*Slightly clarify the wording, which means to refer to calling this method directly.*

```
def resolve_remote(self, uri):
```
```diff
 
-Does not check the store first, but stores the retrieved document in
-the store if :attr:`RefResolver.cache_remote` is True.
+If called directly, does not check the store first, but after
+retrieving the document at the specified URI it will be saved in
+the store if :attr:`cache_remote` is True.
 
```

LABEL: 

## 14. pair 33 — pallets/flask `0f7b3a4f26`
*document `Flask.register_blueprint` arguments closes #1809*

```
def register(self, app, options, first_registration=False):
```
```diff
-Called by :meth:`Flask.register_blueprint` to register a blueprint
-on the application.  This can be overridden to customize the register
-behavior.  Keyword arguments from
-:func:`~flask.Flask.register_blueprint` are directly forwarded to this
-method in the `options` dictionary.
+Called by :meth:`Flask.register_blueprint` to register all views
+and callbacks registered on the blueprint with the application. Creates
+a :class:`.BlueprintSetupState` and calls each :meth:`record` callback
+with it.
+
+:param app: The application this blueprint is being registered with.
+:param options: Keyword arguments forwarded from
+    :meth:`~Flask.register_blueprint`.
+:param first_registration: Whether this is the first time this
+    blueprint has been registered on the application.
```

LABEL: 

## 15. pair 3640 — sqlalchemy/sqlalchemy `948b14b65b`
*-rewrite expire/refresh section*

```
def expire(self, instance, attribute_names=None):
```
```diff
   indicating a subset of attributes to be expired.
+
+.. seealso::
+
+    :ref:`session_expire` - introductory material
+
+    :meth:`.Session.expire`
+
+    :meth:`.Session.refresh`
```

LABEL: 

## 16. pair 868 — pygments/pygments `569eea6ee8`
*Enable Sphinx nitpicky mode and fix warnings (#2403)*

```
def __init__(self, **options):
```
```diff
 command line usage), there are various utility functions
-available to help with that, see `Option processing`_.
+available to help with that, see `Utilities`_.
```

LABEL: 

## 17. pair 144 — pallets/click `13f075c4bd`
*Update documentation following Colorama removal*

```
def _readline_prompt(func: t.Callable[[str], str], text: str, err: bool) -> str:
```
```diff
-Call a prompt function, passing the full prompt on non-Windows so
-readline can handle line editing and cursor positioning correctly.
+Call a prompt function, passing the full prompt so readline can
+handle line editing and cursor positioning correctly.
 
```

LABEL: 

## 18. pair 345 — Python-Markdown/markdown `15acbdec8c`
*Remove lazy_ol keyword. Use sane_lists extension instead.*

```
def __init__(self, **kwargs):
```
```diff
 * smart_emphasis: Treat `_connected_words_` intelligently Default: True
-* lazy_ol: Ignore number of first item of ordered lists. Default: True
```

LABEL: 

## 19. pair 192 — pallets/jinja `065b58fc74`
*Prevent deadlock in LRUCache.setdefault*

```
def __reversed__(self):
```
```diff
-Iterate over the values in the cache dict, oldest items
+Iterate over the keys in the cache dict, oldest items
 coming first.
```

LABEL: 

## 20. pair 1370 — pydata/xarray `876ee4c79e`
*Fix docstring typo in align()*

```
def align(*objects, **kwargs):
```
```diff
      - 'outer': use the union of object indexes
-     - 'outer': use the intersection of object indexes
+     - 'inner': use the intersection of object indexes
      - 'left': use indexes from the first object with each dimension
```

LABEL: 

## 21. pair 3977 — scipy/scipy `8cba6cbbf9`
*add wikipedia reference.*

```
def central_diff_weights(Np, ndiv=1):
```
```diff
 f'(x) = 4x, so f'(3) = 12
+
+References
+----------
+.. [1] https://en.wikipedia.org/wiki/Finite_difference
```

LABEL: 

## 22. pair 233 — python-attrs/attrs `e99a9d1348`
*use fancy :param <type> <name>: syntax in the rest of the attr.ib docstring*

```
def attr(default=NOTHING, validator=None, repr=True, cmp=True, hash=True, init=True, convert=None):
```
```diff
 
-:param validator: :func:`callable` that is called by ``attrs``-generated
+:param callable validator: :func:`callable` that is called by ``attrs``-generated
     ``__init__`` methods after the instance has been initialized.  They
     :func:`get_run_validators`.
-:type validator: callable
 
-:param repr: Include this attribute in the generated ``__repr__`` method.
-:type repr: bool
+:param bool repr: Include this attribute in the generated ``__repr__`` method.
 
-:param cmp: Include this attribute in the generated comparison methods
+:param bool cmp: Include this attribute in the generated comparison methods
     (``__eq__`` et al).
-:type cmp: bool
 
-:param hash: Include this attribute in the generated ``__hash__`` method.
-:type hash: bool
+:param bool hash: Include this attribute in the generated ``__hash__`` method.
 
-:param init: Include this attribute in the generated ``__init__`` method.
-:type init: bool
+:param bool init: Include this attribute in the generated ``__init__`` method.
 
```

LABEL: 

## 23. pair 4092 — scipy/scipy `a62b372b94`
*MAINT: Fixed detection of gimbal lock*

```
def as_euler(self, seq, degrees=False):
```
```diff
         - [-90, 90] degrees if all axes are unique (like xyz)
-        - [0, 180] degrees if all axes are not unique (like zxz)
+        - [0, 180] degrees if first and third axes are same (like zxz)
 
```

LABEL: 

## 24. pair 1718 — statsmodels/statsmodels `8ea3056b3f`
*DOC: Updated DOC on fit()*

```
def fit(self):
```
```diff
-Fits an AFT model and returns parameters.
+Fits an AFT model and returns results instance
 
 -------
-Fitted params
+Results instance.
 
```

LABEL: 

## 25. pair 6888 — pypa/setuptools `16cffb5544`
*Clean up docstring*

```
def write_manifest(self):
```
```diff
-Write the file list in 'self.filelist' (presumably as filled in
-by 'add_defaults()' and 'read_template()') to the manifest file
+Write the file list in 'self.filelist' to the manifest file
 named by 'self.manifest'.
```

LABEL: 

## 26. pair 6834 — pypa/setuptools `55ccc6a11f`
*Add a regression test.*

```
def _is_valid_macro(name, value=None):
```
```diff
 A valid macro is a ``name : str`` and a ``value : str | None``.
+
+>>> Compiler._is_valid_macro('foo', None)
+True
```

LABEL: 

## 27. pair 3338 — django/django `16a842b379`
*Refs #26621 -- Added tests for admindocs.views.simplify_regex().*

```
def simplify_regex(pattern):
```
```diff
-Clean up urlpattern regexes into something somewhat readable by Mere Humans:
-turns something like "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
-into "<sport_slug>/athletes/<athlete_slug>/"
+Clean up urlpattern regexes into something more readable by humans. For
+example, turn "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
+into "/<sport_slug>/athletes/<athlete_slug>/".
```

LABEL: 

## 28. pair 187 — pallets/jinja `e8da6267f8`
*get_template() uses / separators even on Windows*

```
def get_template(self, name: t.Union[str, 'Template'], parent: t.Optional[str]=None, globals: t.Optional[t.MutableMapping[str, t.Any]]=None) -> 'Template':
```
```diff
 
-:param name: Name of the template to load.
+:param name: Name of the template to load. When loading
+    templates from the filesystem, "/" is used as the path
+    separator, even on Windows.
 :param parent: The name of the parent template importing this
```

LABEL: 

## 29. pair 1216 — mwaskom/seaborn `f252b008e9`
*Revamp palette docs and add new functions*

```
def choose_light_palette(input='husl', as_cmap=False):
```
```diff
-Launch an interactive widget to create a sequential palette.
+Launch an interactive widget to create a light sequential palette.
 
```

LABEL: 

## 30. pair 1398 — sphinx-doc/sphinx `1e9108fda6`
*Correct autodoc docstrings*

```
def inherited_members_option(arg: Any) -> Set[str]:
```
```diff
-Used to convert the :members: option to auto directives.
+Used to convert the :inherited-members: option to auto directives.
```

LABEL: 

## 31. pair 330 — encode/httpx `9ae170a936`
*Fix optional percent encoding behaviour. (#2671)*

```
def percent_encode(char: str) -> str:
```
```diff
-Replace every character in a string with the percent-encoded representation.
+Replace a single character with the percent-encoded representation.
 
```

LABEL: 

## 32. pair 1692 — statsmodels/statsmodels `5b44dc0286`
*DOC: Add plot to docstring.*

```
def month_plot(x, dates=None, ylabel=None, ax=None):
```
```diff
 >>> fig = sm.graphics.tsa.month_plot(dta)
+
+.. plot:: plots/graphics_month_plot.py
```

LABEL: 

## 33. pair 339 — facelessuser/pymdown-extensions `995ecae9c0`
*Add a Blocks on_inline_end event*

```
def __init__(self, length, tracker, md, config):
```
```diff
 
-- `store` allows us to store content until all content is found.
-- `length` specifies the length (number of colons) that the header used
+- `length` specifies the length (number of slashes) that the header used
 - `tracker` is a persistent storage for the life of the current Markdown page.
```

LABEL: 

## 34. pair 321 — cpburnz/python-pathspec `101e6284b7`
*Fix doc build*

```
def check_file(self, file: TStrPath, separators: Optional[Collection[str]]=None) -> CheckResult[TStrPath]:
```
```diff
 
-Returns the file check result (:class:`CheckResult`).
+Returns the file check result (:class:`~pathspec.util.CheckResult`).
```

LABEL: 

## 35. pair 220 — python-attrs/attrs `cf0adc5d12`
*Fix typo in `setup.py` (#656)*

```
def read(*parts):
```
```diff
-Build an absolute path from *parts* and and return the contents of the
+Build an absolute path from *parts* and return the contents of the
 resulting file.  Assume UTF-8 encoding.
```

LABEL: 

## 36. pair 737 — networkx/networkx `39aad04536`
*fix typo, add definition and silent the error for isolated nodes.*

```
def overall_reciprocity(G):
```
```diff
-Computer the reciprocity for the whole graph.
+Compute the reciprocity for the whole graph.
+
+See the doc of reciprocity for the definition.
 
```

LABEL: 

## 37. pair 1421 — sphinx-doc/sphinx `40f4db30e8`
*C, remove dead code*

```
def _parse_type(self, named: Union[bool, str], outer: str=None) -> ASTType:
```
```diff
-named=False|'maybe'|True: 'maybe' is e.g., for function objects which
-doesn't need to name the arguments
+named=False|'single'|True: 'single' is e.g., for function objects which
+doesn't need to name the arguments, but otherwise is a single name
```

LABEL: 

## 38. pair 2818 — numpy/numpy `8d3e32cb1f`
*Update*

```
def expandtabs(a, tabsize=8):
```
```diff
 >>> np.char.expandtabs(a, tabsize=8)  # doctest: +SKIP
-array(['                Hello
+array([b'                Hello
     wo'], dtype='|S32')  # doctest: +SKIP
```

LABEL: 

## 39. pair 3358 — django/django `8d90489fe0`
*Fixed typo in escape_filter() docstring.*

```
def escape_filter(value):
```
```diff
-Marks the value as a string that should not be auto-escaped.
+Marks the value as a string that should be auto-escaped.
```

LABEL: 

## 40. pair 3365 — django/django `03467368db`
*Fixed #23558 -- documented slugify limitations*

```
def slugify(value):
```
```diff
-Converts to lowercase, removes non-word characters (alphanumerics and
-underscores) and converts spaces to hyphens. Also strips leading and
-trailing whitespace.
+Converts to ASCII. Converts spaces to hyphens. Removes characters that
+aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
+Also strips leading and trailing whitespace.
```

LABEL: 

## 41. pair 1284 — pydata/xarray `c195c2636a`
*add polyval to polyfit see also (#5020)*

```
def polyfit(self, dim: Hashable, deg: int, skipna: bool=None, rcond: float=None, w: Union[Hashable, Any]=None, full: bool=False, cov: bool=False):
```
```diff
 numpy.polyfit
+numpy.polyval
+xarray.polyval
```

LABEL: 

## 42. pair 323 — cpburnz/python-pathspec `101e6284b7`
*Fix doc build*

```
def check_tree_files(self, root: StrPath, on_error: Optional[Callable[[OSError], None]]=None, follow_links: Optional[bool]=None) -> Iterator[CheckResult[str]]:
```
```diff
 Returns an :class:`~collections.abc.Iterator` yielding each file check
-result (:class:`CheckResult`).
+result (:class:`~pathspec.util.CheckResult`).
```

LABEL: 

## 43. pair 1362 — pydata/xarray `50b9c07270`
*reblock no longer reblocks unspecified dimensions*

```
def reblock(self, blockdims=None, blockshape=None):
```
```diff
 
-If neither blockdims nor blockshape is provided, each array in the
-dataset will be reblocked into a single dask array.
+If neither blockdims nor blockshape is provided for one or more
+dimensions, block sizes along that dimension will not be updated;
+non-dask arrays will be converted into dask arrays with a single block.
 
```

LABEL: 

## 44. pair 156 — pallets/click `183a7f8383`
*document that launch(wait) only works for blocking programs*

```
def launch(url, wait=False, locate=False):
```
```diff
 :param url: URL or filename of the thing to launch.
-:param wait: waits for the program to stop.
+:param wait: Wait for the program to exit before returning. This
+    only works if the launched program blocks. In particular,
+    ``xdg-open`` on Linux does not block.
 :param locate: if this is set to `True` then instead of launching the
```

LABEL: 

## 45. pair 451 — python-jsonschema/jsonschema `05912afc67`
*Axe the sentence about bools, it doesn't apply in general.*

```
def _generate_legacy_type_checks(types=()):
```
```diff
-Generate type check definitions suitable for TypeChecker.redefine_many,
-using the supplied types. Type Checks are simple isinstance checks,
-except checking that numbers aren't really bools.
+Generate newer-style type checks out of JSON-type-name-to-type mappings.
 
 
-        A mapping of type names to their Python Types
+        A mapping of type names to their Python types
 
 
-    A dictionary of definitions to pass to TypeChecker
+    A dictionary of definitions to pass to `TypeChecker`
```

LABEL: 

## 46. pair 4771 — nltk/nltk `7cefb5a4fb`
*added methods to access LICENSE and citation.bib files, cf #31*

```
def abspath(self, fileid):
```
```diff
 
-:type file: str
-:param file: The file identifier for the file whose path
+:type fileid: str
+:param fileid: The file identifier for the file whose path
     should be returned.
```

LABEL: 

## 47. pair 194 — pallets/jinja `996bcadc3e`
*document that sort is stable*

```
def do_sort(environment, value, reverse=False, case_sensitive=False, attribute=None):
```
```diff
 Sort an iterable.  Per default it sorts ascending, if you pass it
-true as first argument it will reverse the sorting.
+true as first argument it will reverse the sorting. The sort is stable,
+i.e. it guarantees not to change the relative order of elements that
+compare equal.
 
 
+Because the sort is stable, it is possible to chain sorts on different
+attributes and ordering:
+
+.. sourcecode:: jinja
+
+    {% for item in iterable|sort(attribute='name')|sort(true,attribute='date') %}
+        ...
+    {% endfor %}
+
 .. versionchanged:: 2.6
    The attribute parameter can contain multiple comma separated
-   attributes, e.g. attr1,attr2.
+   attributes, e.g. 'attr1,attr2', all subject to the same ordering.
```

LABEL: 

## 48. pair 4900 — pydantic/pydantic `8a389f0ffd`
*Improve `pydantic/_internal/_generics.py` docstring (#6151)*

```
def _get_caller_frame_info(depth: int=2) -> tuple[str | None, bool]:
```
```diff
 
-:returns Tuple[module_name, called_globally]
+Args:
+    depth: The depth to get the frame.
+
+Returns:
+    A tuple contains `module_nam` and `called_globally`.
+
+Raises:
+    RuntimeError: If the function is not called inside a function.
```

LABEL: 

## 49. pair 634 — networkx/networkx `6e595bdc84`
*MAINT: Update nx.info (#4193)*

```
def from_pandas_adjacency(df, create_using=None):
```
```diff
 >>> print(nx.info(G))
-Name: Graph from pandas adjacency matrix
-Type: Graph
-Number of nodes: 2
-Number of edges: 3
-Average degree:   3.0000
+Graph named 'Graph from pandas adjacency matrix' with 2 nodes and 3 edges
```

LABEL: 

## 50. pair 4910 — pydantic/pydantic `459b6c675b`
*📝 Update docstings on `networks.py` (#6069)*

```
def validate_email(value: str) -> tuple[str, str]:
```
```diff
 
-Notes:
-* raw ip address (literal) domain parts are not allowed.
-* "John Doe <local_part@domain.com>" style "pretty" email addresses are processed
-* spaces are striped from the beginning and end of addresses but no error is raised
+Note:
+    Note that:
+
+    * Raw IP address (literal) domain parts are not allowed.
+    * "John Doe <local_part@domain.com>" style "pretty" email addresses are processed.
+    * Spaces are striped from the beginning and end of addresses, but no error is raised.
```

LABEL: 
