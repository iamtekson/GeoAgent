# GeoAgent tests

The tests run GeoAgent headless against a real QGIS install. Processing algorithms (buffer, clip, slope, ...) really execute on the bundled demo data; only the LLM is replaced by a scripted stand-in. That makes the suite offline, free, and deterministic: no API key, no Ollama server, no network. The full suite takes about 15 seconds.

## Requirements

- QGIS 3.x or 4.x installed (developed against QGIS 4.0.2), including the GDAL provider.
- GeoAgent's Python dependencies installed into **QGIS's** Python (plugin **Settings** tab → *Check / Install Dependencies*). `langchain-openai` is optional; without it one test is skipped.

The tests use a throw-away QGIS profile, so they never read or write your real QGIS settings, saved API keys, or GeoAgent log.

## Running the tests

Run from the plugin folder (the one containing `metadata.txt`) with the Python that ships with QGIS.

**Windows** (adjust the QGIS version in the path; for an OSGeo4W install use `C:\OSGeo4W\bin\python-qgis.bat`):

```powershell
# PowerShell needs the leading "&"; drop it in cmd.exe
& "C:\Program Files\QGIS 4.0.2\bin\python-qgis.bat" -m unittest discover -s tests -v
```

**Linux / macOS** (any Python that can `import qgis`):

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
```

A passing run ends with:

```
Ran 76 tests in 10.3s

OK
```

Log lines such as `GEO execute failed: Field 'no_such_field' was not found` are expected: the retry tests make steps fail on purpose.

### Running part of the suite

```powershell
# One file
& "C:\Program Files\QGIS 4.0.2\bin\python-qgis.bat" -m unittest discover -s tests -p "test_workflow.py"

# One class or one test: run from inside tests/
cd tests
& "C:\Program Files\QGIS 4.0.2\bin\python-qgis.bat" -m unittest test_workflow.RetryTest
& "C:\Program Files\QGIS 4.0.2\bin\python-qgis.bat" -m unittest -v test_workflow.RetryTest.test_bad_parameter_retries_same_algorithm
```

## What is tested

### `test_parameters.py`: parameter handling (`tools/geoprocessing.py`)

| Test | Checks that |
| --- | --- |
| `NormalizeParametersTest.test_valid_values_pass_through_untouched` | values QGIS already accepts are never rewritten |
| `…test_near_miss_layer_name_resolves_to_layer_id` | `"River"` resolves to the `rivers` layer |
| `…test_raster_parameter_only_matches_raster_layers` | a raster parameter only fuzzy-matches raster layers |
| `…test_enum_label_becomes_index` | option labels (`"Flat"`, `"square"`) become indexes |
| `…test_parameter_name_case_is_fixed` | `"input"` becomes `"INPUT"` |
| `…test_output_label_resolves_to_layer_id` | `task_1_output` / `@task_1_output` resolve to that output's layer |
| `…test_output_name_is_pinned_to_this_runs_output` | a name shared by an old layer and this run's output picks the output |
| `…test_unresolvable_value_is_left_for_validation` | unknown names are left alone, not guessed |
| `…test_unknown_algorithm_returns_input_unchanged` | normalization never fails the run |
| `ValidateParametersTest` (3 tests) | the pre-flight check passes valid input, names every bad value, and reports unknown algorithms |
| `AlgorithmMetadataTest.test_parameter_flags` | advanced/hidden flags are reported per parameter |
| `ExecuteProcessingTest.test_outputs_get_ids_and_unique_names` | results carry layer ids, and repeated runs get unique layer names |

### `test_workflow.py`: the processing workflow (`agents/workflow.py`, `agents/geoprocessing_flow.py`)

| Test | Checks that |
| --- | --- |
| `RoutingAndWiringTest.test_without_flag_routing_falls_back_to_llm_as_before` | without `is_geoprocessing`, the exact old sequence of LLM calls runs |
| `…test_flag_from_decomposition_skips_routing_call` | with the flag, no routing LLM call is made |
| `…test_gather_prompt_lists_advanced_params_and_dependency` | advanced parameters are listed separately, and the dependency hint appears |
| `…test_slips_are_normalized_before_execution` | wrong case, near-miss names, and enum labels still run, wired to the right layers |
| `…test_repeated_run_uses_its_own_output` | a second identical run wires to its own buffer, not the first run's |
| `…test_layer_added_by_llm_task_feeds_dependent_task` | a layer added by an "add layer" step feeds the next step |
| `RetryTest.test_preflight_rejection_regathers_without_llm_analysis` | a QGIS-rejected parameter is re-gathered with no error-analysis LLM call |
| `…test_bad_parameter_retries_same_algorithm` | a "bad parameter" failure keeps the algorithm |
| `…test_wrong_algorithm_reselects` | a "wrong algorithm" failure picks another |
| `…test_failed_analysis_falls_back_to_reselection` | if error analysis itself fails, behaviour is as before (re-select) |
| `…test_same_algorithm_retried_only_once` | a second failure with the same algorithm excludes it |
| `LayerListingTest.test_every_layer_kind_is_listed` | the layer list the LLM sees describes vector (with geometry), raster, geometry-less and other layer kinds |
| `CheckpointTest.test_steps_survive_the_checkpointer` | executed steps can be read back from the saved conversation state |

### `test_model_export.py`: model export (`utils/model_export.py`)

`ModelExportTest` runs one buffer → clip → slope workflow, then checks the exported model:

| Test | Checks that |
| --- | --- |
| `test_steps_collected_in_order` | the three executed steps are collected in task order |
| `test_one_child_per_step` | each step becomes one model algorithm, labelled with its task |
| `test_user_layers_become_model_inputs` | `rivers` and `DEM` become model inputs that default to those layers |
| `test_static_values_kept` | values like the 500 m distance are kept |
| `test_steps_wired_through_child_outputs` | clip's mask comes from buffer's output, and slope's input from clip's |
| `test_only_final_result_is_a_model_output` | intermediate results stay temporary |
| `test_layout_and_validity` | steps are laid out top-down, and QGIS validates every step |
| `test_saved_model_reloads_and_runs` | the `.model3` file reloads and the model runs end to end in QGIS |
| `test_opens_in_model_designer` | QGIS's Model Designer opens it |
| `test_default_models_folder_exists` | the default save folder exists |

`ModelExportEdgeCasesTest` checks that an empty run is an error, that failed and non-processing tasks are skipped, and that a list of input layers becomes one multi-layer input.

### `test_main_thread.py`: running tools on the main thread (`utils/canvas_refresh.py`)

Tools that touch the QGIS project run on the main thread while the LLM works in a background thread.

| Test | Checks that |
| --- | --- |
| `test_tool_result_reaches_worker_with_macos_thread_ids` | a real tool's result reaches the worker even with macOS/Linux-sized thread ids ([issue #62](https://github.com/iamtekson/GeoAgent/issues/62)) |
| `test_function_runs_on_the_main_thread` | the function really runs on the main thread |
| `test_exceptions_reach_the_caller` | a tool's error is raised in the calling thread, with its type and message |
| `test_none_result_is_returned` / `test_arguments_are_passed` | `None` results and arguments come through unchanged |
| `test_call_from_main_thread_runs_directly` | a call made on the main thread runs directly instead of deadlocking |
| `test_without_runner_is_an_error` | calling before the plugin set up its runner is a clear error |

### `test_usage.py`: token usage per request (`llm/usage.py`)

| Test | Checks that |
| --- | --- |
| `TokenUsageTrackerTest` (4 tests) | usage is summed across calls, older providers' `llm_output` counts are read, calls without usage are counted, and the log line reads right in every case |
| `RequestUsageTest.test_processing_request_counts_all_calls` | every LLM call of a processing request is counted, including routing, selection, gathering, error analysis and the summary inside sub-graphs |
| `…test_tool_using_task_counts_all_calls` | tool-calling rounds in a processing task are counted |
| `…test_general_mode_counts_all_calls` | general mode's LLM ⇄ tools loop is counted |
| `…test_provider_without_usage` | a provider that reports no usage gives "not reported by the provider" |

These run the graph the way the plugin's worker thread does (own thread, own event loop), so they also prove that usage tracking reaches nested calls on that path.

### `test_plugin.py`: the plugin itself (`geo_agent.py`, the dialog and `.ui` file)

These use the real `GeoAgent` class, dialog, and background worker thread, with a stub QGIS interface. The numbered tests share one plugin instance and run in order.

| Test | Checks that |
| --- | --- |
| `test_01_chat_bar_has_no_model_buttons` | the chat bar keeps its controls, with no model buttons |
| `test_02_after_run_setting_defaults_to_just_showing` | the Settings tab's *After a processing run* choices, default *Just show the result* |
| `test_03_processing_result_has_model_links` | a processing result ends with *Open in Model Designer · Save as .model3* links; nothing opens by default |
| `test_04_open_link_opens_model_designer` | the open link shows the run in the Model Designer, without errors |
| `test_05_save_link_writes_model3` | the save link writes a `.model3` file (adding the extension) and reports success |
| `test_06_each_link_exports_its_own_run` | links of an earlier run still export that run, not the latest |
| `test_07_setting_opens_designer_after_each_run` | *Open in Model Designer* setting opens the designer automatically |
| `test_08_setting_asks_to_save_after_each_run` | *Ask to save as .model3* setting opens the save dialog automatically |
| `test_09_no_links_without_a_geoprocessing_step` | a run with nothing to export gets no links |
| `test_10_general_mode_unaffected` | general mode answers as before, without links |
| `test_11_token_usage_logged_after_each_request` | each request ends its log with the tokens it used and the session total |
| `test_12_web_links_open_in_browser` | ordinary links in replies open in the web browser |
| `test_13_clear_chat_forgets_runs` | Clear Chat forgets the runs and their links |
| `test_14_after_run_setting_is_saved` | *Save settings* keeps the *After a processing run* choice |
| `test_15_first_message_builds_agent_once` | the first message builds the LLM client and graph once, not twice |
| `test_16_any_changed_setting_rebuilds_agent` | changing max tokens, model name, API key or an Ollama setting applies to the next message; unchanged settings don't rebuild |
| `test_17_initialize_agent_builds_both_modes` | the real provider setup builds both modes (skipped without `langchain-openai`) |
| `test_18_error_log_falls_back_when_file_is_unwritable` | if the error-log file can't be written, the error still reaches the QGIS log |
| `test_19_corrupted_saved_setting_does_not_block_the_panel` | a corrupted saved value is logged and the default kept; the panel still opens |
| `test_99_unload_twice_is_safe` | unloading the plugin twice doesn't fail |
| `UILogHandlerTest.test_deleted_logs_tab_is_dropped_not_crashed_on` | after the Logs tab is deleted (plugin unloaded), logging carries on without errors |

## How it works

- **`_bootstrap.py`**: every test module imports it first. It:
  - starts one offscreen `QgsApplication` with a temporary profile and initializes Processing;
  - makes the plugin importable as `geo_agent`, whatever the folder is called;
  - copies the demo layers to a temp folder: `rivers.shp` (lines) and `DEM.tif` (raster), both EPSG:32644.

  `reset_project()` gives each test a fresh project with just those two layers.
- **`fake_llm.py`**: `FakeLLM` is a real LangChain chat model (`BaseChatModel`). Structured output, tool binding and callbacks such as token tracking go through the same LangChain code as with a real provider; only the answers are scripted.
  - **Scripted answers.** Each structured-output call is answered from a script keyed by schema name (`"TaskDecomposition"`, `"AlgorithmSelection"`, `"ParameterGathering"`, `"ErrorAnalysis"`, `"RouteDecision"`). An answer is a dict of field values, a function of the prompt messages, or an exception to raise.
  - **Plain calls.** Tool-using tasks and the summary get the `tool_calls=` script, then a fixed summary.
  - **Usage.** Every response reports `USAGE` tokens (100 in, 20 out); pass `usage=None` for a provider that reports none.
  - **Call log.** It records every call, so tests can check which LLM calls happened (`llm.names()`, `llm.count(...)`) and what the prompts said (`llm.prompts(...)`).
  - **Helpers.** `task()`, `decomposition()`, `selection()` and `params()` keep scripts short.

## Writing a new test

```python
import unittest

import _bootstrap  # first: starts QGIS and makes `geo_agent` importable
from fake_llm import FakeLLM, decomposition, params, selection, task
from langchain_core.messages import HumanMessage

from geo_agent.agents.workflow import build_workflow_graph


class DissolveTest(unittest.TestCase):
    def setUp(self):
        _bootstrap.reset_project()  # layers: "rivers", "DEM"

    def test_dissolve_rivers(self):
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(task(1, "Dissolve rivers", "dissolve", ["dissolve"], geo=True))],
            "AlgorithmSelection": [selection("native:dissolve")],
            "ParameterGathering": [params(INPUT="rivers")],
        })
        app = build_workflow_graph(llm).compile()
        out = app.invoke({"messages": [HumanMessage(content="dissolve rivers")]}, {"recursion_limit": 100})
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])
```

Guidelines:

- Keep tests offline: script the LLM; never call a real provider.
- Write any files to a `tempfile.mkdtemp()` folder and remove it afterwards.
- Name test files `test_*.py` so discovery finds them.
- Code that opens a modal dialog (`QMessageBox`, `exec()`) blocks headless runs: stub it, as `test_plugin.py` does with `showMessage`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'qgis'` | Use the Python that ships with QGIS (`python-qgis.bat` on Windows). |
| `ModuleNotFoundError: langgraph` (or another package) | Install the plugin's dependencies into QGIS's Python (Settings tab → *Check / Install Dependencies*). |
| `could not connect to display` (Linux) | Make sure `QT_QPA_PLATFORM=offscreen` is set; `_bootstrap.py` sets it unless the variable already exists. |
| A run hangs | A modal dialog opened somewhere; stub it in the test. |
| `ModuleNotFoundError: No module named 'test_…'` when running a single test | Run single classes or tests from inside `tests/` (see above). |

The tests are not part of the plugin release: `package_plugin.py` excludes the `tests/` folder from the zip.
