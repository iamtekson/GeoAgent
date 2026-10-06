# Contributing

Contributions are welcome: fork the [repository](https://github.com/iamtekson/GeoAgent), create a feature branch, and open a pull request. For bugs, [open an issue](https://github.com/iamtekson/GeoAgent/issues).

## Development setup

Clone the repository straight into your QGIS plugins folder, under the name `geo_agent`:

```bash
# Windows, QGIS 4 (use QGIS3 for QGIS 3)
cd "%APPDATA%\QGIS\QGIS4\profiles\default\python\plugins"
# Linux:  cd ~/.local/share/QGIS/QGIS4/profiles/default/python/plugins
# macOS:  cd ~/Library/Application\ Support/QGIS/QGIS4/profiles/default/python/plugins
git clone https://github.com/iamtekson/GeoAgent.git geo_agent
```

Restart QGIS, enable GeoAgent in the Plugin Manager, and install its dependencies from the Settings tab. The [Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/) plugin reloads GeoAgent after code changes without restarting QGIS.

See [Architecture](architecture.md) for how the code is organized.

## Before opening a pull request

- Run the [tests](testing.md); they take about 15 seconds and need no API key.
- Add tests for new behaviour. The scripted LLM in `tests/fake_llm.py` makes most flows testable offline.
- Run the [Bandit](https://bandit.readthedocs.io/) security scan, which the QGIS plugin repository also runs on uploads. It should report no issues:

  ```bash
  uvx bandit -r . -x ./tests,./docs,./paper,./_extra
  ```

  Avoid `except Exception: pass`: catch the specific error you expect and handle or log it. If a finding is reviewed and safe, mark the line with `# nosec <test id>` and a comment explaining why.
- Update the documentation in `docs/` if users will notice the change.

## Documentation

The documentation is written in Markdown ([MyST](https://myst-parser.readthedocs.io/)) and built with Sphinx. It is published on Read the Docs from the `docs/` folder; `.readthedocs.yaml` holds the build configuration. To build it locally:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`. The `-W` flag treats warnings as errors, as the Read the Docs build does.

### Screenshots

The screenshots in `docs/imgs/screenshots/` are rendered from the real plugin UI. `docs/scripts/take_screenshots.py` runs a processing request on the demo data and renders the panel tabs, the Model Designer and a map of the results. The LLM's replies are scripted; everything else is real. Regenerate them after UI changes, with QGIS's Python, from the plugin folder:

```bash
# Windows (adjust the QGIS version in the path)
"C:\Program Files\QGIS 4.0.2\bin\python-qgis.bat" docs/scripts/take_screenshots.py

# Linux / macOS (Python that can `import qgis`)
python3 docs/scripts/take_screenshots.py
```

## Packaging a release

1. Update the version in `metadata.txt`; the documentation reads its version from there. Update `pyproject.toml` to match.
2. Add the changes to the changelog in `metadata.txt` and in `docs/changelog.md`.
3. Build the plugin zip:

   ```bash
   python package_plugin.py
   ```

   This writes `geo_agent-<version>.zip` next to the plugin folder. It leaves out development files: tests, docs, the paper, caches. Use `--output` to choose another location, or `--no-version` to drop the version from the file name.
4. Install the zip in a clean QGIS profile to check it, then upload it to [plugins.qgis.org](https://plugins.qgis.org/).
