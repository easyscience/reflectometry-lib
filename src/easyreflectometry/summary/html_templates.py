# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

HTML_TEMPLATE = """<!DOCTYPE html>

<html>

    <style>
        th, td {
            padding-right: 18px;
        }
        th {
            text-align: left;
        }
    </style>

    <body>
        <table>

            <!-- Summary title -->
            <tr>
                <td><h1>Summary</h1></td>
            </tr>

            <!-- Project -->
            project_information_section

            <tr></tr>

            <!-- Sample -->
            <tr>
                <td><h2>Sample</h2></td>
            </tr>
            sample_section

            <tr></tr>

            <!-- Experiments -->
            <tr>
                <td><h2>Experiments</h2></td>
            </tr>
            experiments_section

            <tr></tr>

            <!-- Analysis -->
            <tr>
                <td><h2>Refinement</h2></td>
            </tr>
            refinement_section

        </table>

        figures_section

    </body>
</html>"""


HTML_PROJECT_INFORMATION_TEMPLATE = """
<tr>
    <td><h3>Project information</h3></td>
</tr>

<tr>
    <th>Title</th>
    <th>project_title</th>
</tr>
<tr>
    <td>Description</td>
    <td>project_description</td>
</tr>
<tr>
    <td>No. of experiments</td>
    <td>num_experiments</td>
</tr>
"""

HTML_PARAMETER_HEADER_TEMPLATE = """
<tr>
    <th>parameter_name</th>
    <th>parameter_value</th>
    <th>parameter_unit</th>
    <th>parameter_error</th>
</tr>
"""

HTML_PARAMETER_TEMPLATE = """
<tr>
    <td>parameter_name</td>
    <td>parameter_value</td>
    <td>parameter_unit</td>
    <td>parameter_error</td>
</tr>
"""

HTML_DATA_COLLECTION_TEMPLATE = """
<tr>
    <th>Experiment datablock</th>
    <th>experiment_name</th>
</tr>
<tr>
    <td>Measured intensity range:</td>
    <td>[range_min, range_max]</td>
</tr>
<tr>
    <td>No. of data points</td>
    <td>num_data_points</td>
</tr>
<tr>
    <td>Resolution function</td>
    <td>resolution_function</td>
</tr>

"""

HTML_REFINEMENT_TEMPLATE = """
<tr>
    <td>Calculation engine</td>
    <td>calculation_engine</td>
</tr>
<tr>
    <td>Minimization engine</td>
    <td>minimization_engine</td>
</tr>
<tr>
    <td>Goodness-of-fit: reduced <i>&chi;</i><sup>2</sup></td>
    <td>goodness_of_fit</td>
</tr>
<tr>
    <td>No. of free parameters in the fit:</td>
    <td>num_fit_free_params</td>
</tr>
<tr>
    <td>No. of parameters (all models):</td>
    <td>num_total_params</td>
</tr>
<tr>
    <td>No. of free parameters (all models):</td>
    <td>num_free_params</td>
</tr>
<tr>
    <td>No. of fixed parameters (all models):</td>
    <td>num_fixed_params</td>
</tr>
<tr>
    <td>No. of constraints (all models):</td>
    <td>num_constriants</td>
</tr>
"""

HTML_FIGURES_TEMPLATE = """
<img src="path_sld_plot" alt="SLD plot" width="640" height="480">
<br>
<img src="path_fit_experiment_plot" alt="Fit experiment plot" width="640" height="480">
"""

# Interactive figures for HTML reports. The plotly ``<div>``s carry their own
# JavaScript so the report stays interactive (zoom, pan, hover) when opened in a
# browser. The plotly.js library is embedded inline in the first figure, which
# keeps the saved report self-contained and working offline.
HTML_INTERACTIVE_FIGURES_TEMPLATE = """
sld_plot_div
<br>
fit_experiment_plot_div
"""
