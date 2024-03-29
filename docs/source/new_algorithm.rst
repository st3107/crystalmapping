Use new algorithm to run the batch processing
=============================================

Create the calculator
---------------------

All the data processing is done by the ``Calculator``. The input data
and output data are stored in its attributes.

.. code:: ipython3

    from crystalmapping.utils import Calculator

    calculator = Calculator()

Load the experiment Data
------------------------

In this example, we load the data from a database. You can also use your
data source as long as there is an ``DataArray`` of the exposure images.

.. code:: ipython3

    from databroker import catalog

    list(catalog)




.. parsed-literal::

    ['test_data_in_database',
     'analysis',
     'bt_safN_306132',
     'pdf',
     'saf_307381',
     'xpd']



.. code:: ipython3

    db = catalog["xpd"]

.. code:: ipython3

    UID = '257b5581-ca78-4309-9c50-b4d65d80152a'
    run = db[UID]
    run




.. parsed-literal::

    BlueskyRun
      uid='257b5581-ca78-4309-9c50-b4d65d80152a'
      exit_status='success'
      2021-03-19 22:48:19.253 -- 2021-03-19 23:13:41.753
      Streams:
        * primary




.. code:: ipython3

    data = run.primary.to_dask()
    data




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */

    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }

    html[theme=dark],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1F1F1F;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }

    .xr-wrap {
      display: block;
      min-width: 300px;
      max-width: 700px;
    }

    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }

    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }

    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }

    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }

    .xr-obj-type {
      color: var(--xr-font-color2);
    }

    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }

    .xr-section-item {
      display: contents;
    }

    .xr-section-item input {
      display: none;
    }

    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }

    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }

    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }

    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }

    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }

    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }

    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }

    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }

    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }

    .xr-section-summary-in:checked + label > span {
      display: none;
    }

    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }

    .xr-section-inline-details {
      grid-column: 2 / -1;
    }

    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }

    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }

    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }

    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }

    .xr-preview {
      color: var(--xr-font-color3);
    }

    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }

    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }

    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }

    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }

    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }

    .xr-dim-list:before {
      content: '(';
    }

    .xr-dim-list:after {
      content: ')';
    }

    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }

    .xr-has-index {
      font-weight: bold;
    }

    .xr-var-list,
    .xr-var-item {
      display: contents;
    }

    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }

    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }

    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }

    .xr-var-name {
      grid-column: 1;
    }

    .xr-var-dims {
      grid-column: 2;
    }

    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }

    .xr-var-preview {
      grid-column: 4;
    }

    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }

    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }

    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }

    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }

    .xr-var-data > table {
      float: right;
    }

    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }

    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }

    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }

    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }

    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }

    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }

    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }

    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
    Dimensions:              (dim_0: 1, dim_1: 3888, dim_2: 3072, time: 1001)
    Coordinates:
      * time                 (time) float64 1.616e+09 1.616e+09 ... 1.616e+09
    Dimensions without coordinates: dim_0, dim_1, dim_2
    Data variables:
        dexela_stats1_total  (time) float64 dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;
        dexela_image         (time, dim_0, dim_1, dim_2) float64 dask.array&lt;chunksize=(1, 1, 3888, 3072), meta=np.ndarray&gt;
        mPhi                 (time) float64 dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;
        mPhi_user_setpoint   (time) float64 dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-1a568b70-e659-4d38-9311-4d88b43c7155' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1a568b70-e659-4d38-9311-4d88b43c7155' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span>dim_0</span>: 1</li><li><span>dim_1</span>: 3888</li><li><span>dim_2</span>: 3072</li><li><span class='xr-has-index'>time</span>: 1001</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-c9e8d875-af9a-4879-8ed0-0bbf26d84d62' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c9e8d875-af9a-4879-8ed0-0bbf26d84d62' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.616e+09 1.616e+09 ... 1.616e+09</div><input id='attrs-3f8d8141-c3fb-470c-8e4a-0e0daca15790' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3f8d8141-c3fb-470c-8e4a-0e0daca15790' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0b7b72db-453e-4c96-a0eb-ce82e32637bf' class='xr-var-data-in' type='checkbox'><label for='data-0b7b72db-453e-4c96-a0eb-ce82e32637bf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.616209e+09, 1.616209e+09, 1.616209e+09, ..., 1.616210e+09,
           1.616210e+09, 1.616210e+09])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-54739c24-e6d9-4da8-9636-c053e372f34f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54739c24-e6d9-4da8-9636-c053e372f34f' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>dexela_stats1_total</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;</div><input id='attrs-0eddac4b-eb1d-44fd-88b5-3b7b57208fd5' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-0eddac4b-eb1d-44fd-88b5-3b7b57208fd5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78e05870-4ed0-4a50-a455-c002eae84dba' class='xr-var-data-in' type='checkbox'><label for='data-78e05870-4ed0-4a50-a455-c002eae84dba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>object :</span></dt><dd>dexela</dd></dl></div><div class='xr-var-data'><table>
    <tr>
    <td>
    <table>
      <thead>
        <tr><td> </td><th> Array </th><th> Chunk </th></tr>
      </thead>
      <tbody>
        <tr><th> Bytes </th><td> 7.82 kiB </td> <td> 8 B </td></tr>
        <tr><th> Shape </th><td> (1001,) </td> <td> (1,) </td></tr>
        <tr><th> Count </th><td> 2002 Tasks </td><td> 1001 Chunks </td></tr>
        <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
      </tbody>
    </table>
    </td>
    <td>
    <svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

      <!-- Horizontal lines -->
      <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
      <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
      <line x1="6" y1="0" x2="6" y2="25" />
      <line x1="12" y1="0" x2="12" y2="25" />
      <line x1="18" y1="0" x2="18" y2="25" />
      <line x1="25" y1="0" x2="25" y2="25" />
      <line x1="31" y1="0" x2="31" y2="25" />
      <line x1="37" y1="0" x2="37" y2="25" />
      <line x1="44" y1="0" x2="44" y2="25" />
      <line x1="50" y1="0" x2="50" y2="25" />
      <line x1="56" y1="0" x2="56" y2="25" />
      <line x1="63" y1="0" x2="63" y2="25" />
      <line x1="69" y1="0" x2="69" y2="25" />
      <line x1="75" y1="0" x2="75" y2="25" />
      <line x1="81" y1="0" x2="81" y2="25" />
      <line x1="88" y1="0" x2="88" y2="25" />
      <line x1="94" y1="0" x2="94" y2="25" />
      <line x1="100" y1="0" x2="100" y2="25" />
      <line x1="107" y1="0" x2="107" y2="25" />
      <line x1="113" y1="0" x2="113" y2="25" />
      <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#8B4903A0;stroke-width:0"/>

      <!-- Text -->
      <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >1001</text>
      <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
    </svg>
    </td>
    </tr>
    </table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>dexela_image</span></div><div class='xr-var-dims'>(time, dim_0, dim_1, dim_2)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(1, 1, 3888, 3072), meta=np.ndarray&gt;</div><input id='attrs-643f5159-9556-4970-8576-5536ba3c94f8' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-643f5159-9556-4970-8576-5536ba3c94f8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9a9eff0f-d5b0-4508-937c-a4f533159483' class='xr-var-data-in' type='checkbox'><label for='data-9a9eff0f-d5b0-4508-937c-a4f533159483' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>object :</span></dt><dd>dexela</dd></dl></div><div class='xr-var-data'><table>
    <tr>
    <td>
    <table>
      <thead>
        <tr><td> </td><th> Array </th><th> Chunk </th></tr>
      </thead>
      <tbody>
        <tr><th> Bytes </th><td> 89.08 GiB </td> <td> 91.12 MiB </td></tr>
        <tr><th> Shape </th><td> (1001, 1, 3888, 3072) </td> <td> (1, 1, 3888, 3072) </td></tr>
        <tr><th> Count </th><td> 3003 Tasks </td><td> 1001 Chunks </td></tr>
        <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
      </tbody>
    </table>
    </td>
    <td>
    <svg width="385" height="184" style="stroke:rgb(0,0,0);stroke-width:1" >

      <!-- Horizontal lines -->
      <line x1="0" y1="0" x2="43" y2="0" style="stroke-width:2" />
      <line x1="0" y1="25" x2="43" y2="25" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
      <line x1="1" y1="0" x2="1" y2="25" />
      <line x1="2" y1="0" x2="2" y2="25" />
      <line x1="3" y1="0" x2="3" y2="25" />
      <line x1="5" y1="0" x2="5" y2="25" />
      <line x1="6" y1="0" x2="6" y2="25" />
      <line x1="8" y1="0" x2="8" y2="25" />
      <line x1="9" y1="0" x2="9" y2="25" />
      <line x1="10" y1="0" x2="10" y2="25" />
      <line x1="12" y1="0" x2="12" y2="25" />
      <line x1="13" y1="0" x2="13" y2="25" />
      <line x1="14" y1="0" x2="14" y2="25" />
      <line x1="16" y1="0" x2="16" y2="25" />
      <line x1="17" y1="0" x2="17" y2="25" />
      <line x1="18" y1="0" x2="18" y2="25" />
      <line x1="20" y1="0" x2="20" y2="25" />
      <line x1="21" y1="0" x2="21" y2="25" />
      <line x1="22" y1="0" x2="22" y2="25" />
      <line x1="24" y1="0" x2="24" y2="25" />
      <line x1="25" y1="0" x2="25" y2="25" />
      <line x1="26" y1="0" x2="26" y2="25" />
      <line x1="28" y1="0" x2="28" y2="25" />
      <line x1="29" y1="0" x2="29" y2="25" />
      <line x1="30" y1="0" x2="30" y2="25" />
      <line x1="32" y1="0" x2="32" y2="25" />
      <line x1="33" y1="0" x2="33" y2="25" />
      <line x1="34" y1="0" x2="34" y2="25" />
      <line x1="36" y1="0" x2="36" y2="25" />
      <line x1="37" y1="0" x2="37" y2="25" />
      <line x1="38" y1="0" x2="38" y2="25" />
      <line x1="40" y1="0" x2="40" y2="25" />
      <line x1="41" y1="0" x2="41" y2="25" />
      <line x1="43" y1="0" x2="43" y2="25" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="0.0,0.0 43.0078215732437,0.0 43.0078215732437,25.412616514582485 0.0,25.412616514582485" style="fill:#8B4903A0;stroke-width:0"/>

      <!-- Text -->
      <text x="21.503911" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >1001</text>
      <text x="63.007822" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,63.007822,12.706308)">1</text>


      <!-- Horizontal lines -->
      <line x1="113" y1="0" x2="127" y2="14" style="stroke-width:2" />
      <line x1="113" y1="120" x2="127" y2="134" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="113" y1="0" x2="113" y2="120" style="stroke-width:2" />
      <line x1="127" y1="14" x2="127" y2="134" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="113.0,0.0 127.9485979497544,14.948597949754403 127.9485979497544,134.9485979497544 113.0,120.0" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Horizontal lines -->
      <line x1="113" y1="0" x2="207" y2="0" style="stroke-width:2" />
      <line x1="127" y1="14" x2="222" y2="14" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="113" y1="0" x2="127" y2="14" style="stroke-width:2" />
      <line x1="207" y1="0" x2="222" y2="14" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="113.0,0.0 207.81481481481484,0.0 222.76341276456924,14.948597949754403 127.9485979497544,14.948597949754403" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Horizontal lines -->
      <line x1="127" y1="14" x2="222" y2="14" style="stroke-width:2" />
      <line x1="127" y1="134" x2="222" y2="134" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="127" y1="14" x2="127" y2="134" style="stroke-width:2" />
      <line x1="222" y1="14" x2="222" y2="134" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="127.9485979497544,14.948597949754403 222.76341276456924,14.948597949754403 222.76341276456924,134.9485979497544 127.9485979497544,134.9485979497544" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Text -->
      <text x="175.356005" y="154.948598" font-size="1.0rem" font-weight="100" text-anchor="middle" >3072</text>
      <text x="242.763413" y="74.948598" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,242.763413,74.948598)">3888</text>
      <text x="110.474299" y="147.474299" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,110.474299,147.474299)">1</text>
    </svg>
    </td>
    </tr>
    </table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mPhi</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;</div><input id='attrs-ab69d814-896a-4377-9594-c4b0b425a632' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-ab69d814-896a-4377-9594-c4b0b425a632' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ed4f838b-e34e-423f-9cf1-04c6e52d92e8' class='xr-var-data-in' type='checkbox'><label for='data-ed4f838b-e34e-423f-9cf1-04c6e52d92e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>object :</span></dt><dd>mPhi</dd></dl></div><div class='xr-var-data'><table>
    <tr>
    <td>
    <table>
      <thead>
        <tr><td> </td><th> Array </th><th> Chunk </th></tr>
      </thead>
      <tbody>
        <tr><th> Bytes </th><td> 7.82 kiB </td> <td> 8 B </td></tr>
        <tr><th> Shape </th><td> (1001,) </td> <td> (1,) </td></tr>
        <tr><th> Count </th><td> 2002 Tasks </td><td> 1001 Chunks </td></tr>
        <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
      </tbody>
    </table>
    </td>
    <td>
    <svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

      <!-- Horizontal lines -->
      <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
      <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
      <line x1="6" y1="0" x2="6" y2="25" />
      <line x1="12" y1="0" x2="12" y2="25" />
      <line x1="18" y1="0" x2="18" y2="25" />
      <line x1="25" y1="0" x2="25" y2="25" />
      <line x1="31" y1="0" x2="31" y2="25" />
      <line x1="37" y1="0" x2="37" y2="25" />
      <line x1="44" y1="0" x2="44" y2="25" />
      <line x1="50" y1="0" x2="50" y2="25" />
      <line x1="56" y1="0" x2="56" y2="25" />
      <line x1="63" y1="0" x2="63" y2="25" />
      <line x1="69" y1="0" x2="69" y2="25" />
      <line x1="75" y1="0" x2="75" y2="25" />
      <line x1="81" y1="0" x2="81" y2="25" />
      <line x1="88" y1="0" x2="88" y2="25" />
      <line x1="94" y1="0" x2="94" y2="25" />
      <line x1="100" y1="0" x2="100" y2="25" />
      <line x1="107" y1="0" x2="107" y2="25" />
      <line x1="113" y1="0" x2="113" y2="25" />
      <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#8B4903A0;stroke-width:0"/>

      <!-- Text -->
      <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >1001</text>
      <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
    </svg>
    </td>
    </tr>
    </table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mPhi_user_setpoint</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(1,), meta=np.ndarray&gt;</div><input id='attrs-52fd555f-75e9-4a9a-85fd-3e1c4cda1306' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-52fd555f-75e9-4a9a-85fd-3e1c4cda1306' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c43b1325-5343-4895-a2f5-44fae483166a' class='xr-var-data-in' type='checkbox'><label for='data-c43b1325-5343-4895-a2f5-44fae483166a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>object :</span></dt><dd>mPhi</dd></dl></div><div class='xr-var-data'><table>
    <tr>
    <td>
    <table>
      <thead>
        <tr><td> </td><th> Array </th><th> Chunk </th></tr>
      </thead>
      <tbody>
        <tr><th> Bytes </th><td> 7.82 kiB </td> <td> 8 B </td></tr>
        <tr><th> Shape </th><td> (1001,) </td> <td> (1,) </td></tr>
        <tr><th> Count </th><td> 2002 Tasks </td><td> 1001 Chunks </td></tr>
        <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
      </tbody>
    </table>
    </td>
    <td>
    <svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

      <!-- Horizontal lines -->
      <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
      <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
      <line x1="6" y1="0" x2="6" y2="25" />
      <line x1="12" y1="0" x2="12" y2="25" />
      <line x1="18" y1="0" x2="18" y2="25" />
      <line x1="25" y1="0" x2="25" y2="25" />
      <line x1="31" y1="0" x2="31" y2="25" />
      <line x1="37" y1="0" x2="37" y2="25" />
      <line x1="44" y1="0" x2="44" y2="25" />
      <line x1="50" y1="0" x2="50" y2="25" />
      <line x1="56" y1="0" x2="56" y2="25" />
      <line x1="63" y1="0" x2="63" y2="25" />
      <line x1="69" y1="0" x2="69" y2="25" />
      <line x1="75" y1="0" x2="75" y2="25" />
      <line x1="81" y1="0" x2="81" y2="25" />
      <line x1="88" y1="0" x2="88" y2="25" />
      <line x1="94" y1="0" x2="94" y2="25" />
      <line x1="100" y1="0" x2="100" y2="25" />
      <line x1="107" y1="0" x2="107" y2="25" />
      <line x1="113" y1="0" x2="113" y2="25" />
      <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#8B4903A0;stroke-width:0"/>

      <!-- Text -->
      <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >1001</text>
      <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
    </svg>
    </td>
    </tr>
    </table></div></li></ul></div></li><li class='xr-section-item'><input id='section-eb870b3e-8a5f-4090-a169-e9168ca83d40' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-eb870b3e-8a5f-4090-a169-e9168ca83d40' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Here, we give the data to the attribute.

.. code:: ipython3

    calculator.frames_arr = data["dexela_image"][::10]
    calculator.frames_arr




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */

    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }

    html[theme=dark],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1F1F1F;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }

    .xr-wrap {
      display: block;
      min-width: 300px;
      max-width: 700px;
    }

    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }

    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }

    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }

    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }

    .xr-obj-type {
      color: var(--xr-font-color2);
    }

    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }

    .xr-section-item {
      display: contents;
    }

    .xr-section-item input {
      display: none;
    }

    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }

    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }

    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }

    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }

    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }

    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }

    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }

    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }

    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }

    .xr-section-summary-in:checked + label > span {
      display: none;
    }

    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }

    .xr-section-inline-details {
      grid-column: 2 / -1;
    }

    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }

    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }

    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }

    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }

    .xr-preview {
      color: var(--xr-font-color3);
    }

    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }

    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }

    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }

    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }

    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }

    .xr-dim-list:before {
      content: '(';
    }

    .xr-dim-list:after {
      content: ')';
    }

    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }

    .xr-has-index {
      font-weight: bold;
    }

    .xr-var-list,
    .xr-var-item {
      display: contents;
    }

    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }

    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }

    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }

    .xr-var-name {
      grid-column: 1;
    }

    .xr-var-dims {
      grid-column: 2;
    }

    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }

    .xr-var-preview {
      grid-column: 4;
    }

    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }

    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }

    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }

    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }

    .xr-var-data > table {
      float: right;
    }

    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }

    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }

    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }

    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }

    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }

    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }

    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }

    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;dexela_image&#x27; (time: 101, dim_0: 1, dim_1: 3888, dim_2: 3072)&gt;
    dask.array&lt;getitem, shape=(101, 1, 3888, 3072), dtype=float64, chunksize=(1, 1, 3888, 3072), chunktype=numpy.ndarray&gt;
    Coordinates:
      * time     (time) float64 1.616e+09 1.616e+09 ... 1.616e+09 1.616e+09
    Dimensions without coordinates: dim_0, dim_1, dim_2
    Attributes:
        object:   dexela</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'dexela_image'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 101</li><li><span>dim_0</span>: 1</li><li><span>dim_1</span>: 3888</li><li><span>dim_2</span>: 3072</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-7a74582d-0903-4458-940c-56241d7e5c92' class='xr-array-in' type='checkbox' checked><label for='section-7a74582d-0903-4458-940c-56241d7e5c92' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>dask.array&lt;chunksize=(1, 1, 3888, 3072), meta=np.ndarray&gt;</span></div><div class='xr-array-data'><table>
    <tr>
    <td>
    <table>
      <thead>
        <tr><td> </td><th> Array </th><th> Chunk </th></tr>
      </thead>
      <tbody>
        <tr><th> Bytes </th><td> 8.99 GiB </td> <td> 91.12 MiB </td></tr>
        <tr><th> Shape </th><td> (101, 1, 3888, 3072) </td> <td> (1, 1, 3888, 3072) </td></tr>
        <tr><th> Count </th><td> 3104 Tasks </td><td> 101 Chunks </td></tr>
        <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
      </tbody>
    </table>
    </td>
    <td>
    <svg width="359" height="184" style="stroke:rgb(0,0,0);stroke-width:1" >

      <!-- Horizontal lines -->
      <line x1="0" y1="0" x2="30" y2="0" style="stroke-width:2" />
      <line x1="0" y1="25" x2="30" y2="25" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
      <line x1="0" y1="0" x2="0" y2="25" />
      <line x1="1" y1="0" x2="1" y2="25" />
      <line x1="2" y1="0" x2="2" y2="25" />
      <line x1="3" y1="0" x2="3" y2="25" />
      <line x1="4" y1="0" x2="4" y2="25" />
      <line x1="5" y1="0" x2="5" y2="25" />
      <line x1="6" y1="0" x2="6" y2="25" />
      <line x1="7" y1="0" x2="7" y2="25" />
      <line x1="8" y1="0" x2="8" y2="25" />
      <line x1="9" y1="0" x2="9" y2="25" />
      <line x1="10" y1="0" x2="10" y2="25" />
      <line x1="11" y1="0" x2="11" y2="25" />
      <line x1="12" y1="0" x2="12" y2="25" />
      <line x1="13" y1="0" x2="13" y2="25" />
      <line x1="14" y1="0" x2="14" y2="25" />
      <line x1="15" y1="0" x2="15" y2="25" />
      <line x1="16" y1="0" x2="16" y2="25" />
      <line x1="16" y1="0" x2="16" y2="25" />
      <line x1="17" y1="0" x2="17" y2="25" />
      <line x1="19" y1="0" x2="19" y2="25" />
      <line x1="19" y1="0" x2="19" y2="25" />
      <line x1="20" y1="0" x2="20" y2="25" />
      <line x1="21" y1="0" x2="21" y2="25" />
      <line x1="22" y1="0" x2="22" y2="25" />
      <line x1="23" y1="0" x2="23" y2="25" />
      <line x1="24" y1="0" x2="24" y2="25" />
      <line x1="25" y1="0" x2="25" y2="25" />
      <line x1="26" y1="0" x2="26" y2="25" />
      <line x1="27" y1="0" x2="27" y2="25" />
      <line x1="28" y1="0" x2="28" y2="25" />
      <line x1="29" y1="0" x2="29" y2="25" />
      <line x1="30" y1="0" x2="30" y2="25" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="0.0,0.0 30.53617197430218,0.0 30.53617197430218,25.412616514582485 0.0,25.412616514582485" style="fill:#8B4903A0;stroke-width:0"/>

      <!-- Text -->
      <text x="15.268086" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >101</text>
      <text x="50.536172" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,50.536172,12.706308)">1</text>


      <!-- Horizontal lines -->
      <line x1="100" y1="0" x2="114" y2="14" style="stroke-width:2" />
      <line x1="100" y1="120" x2="114" y2="134" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="100" y1="0" x2="100" y2="120" style="stroke-width:2" />
      <line x1="114" y1="14" x2="114" y2="134" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="100.0,0.0 114.9485979497544,14.948597949754403 114.9485979497544,134.9485979497544 100.0,120.0" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Horizontal lines -->
      <line x1="100" y1="0" x2="194" y2="0" style="stroke-width:2" />
      <line x1="114" y1="14" x2="209" y2="14" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="100" y1="0" x2="114" y2="14" style="stroke-width:2" />
      <line x1="194" y1="0" x2="209" y2="14" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="100.0,0.0 194.81481481481484,0.0 209.76341276456924,14.948597949754403 114.9485979497544,14.948597949754403" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Horizontal lines -->
      <line x1="114" y1="14" x2="209" y2="14" style="stroke-width:2" />
      <line x1="114" y1="134" x2="209" y2="134" style="stroke-width:2" />

      <!-- Vertical lines -->
      <line x1="114" y1="14" x2="114" y2="134" style="stroke-width:2" />
      <line x1="209" y1="14" x2="209" y2="134" style="stroke-width:2" />

      <!-- Colored Rectangle -->
      <polygon points="114.9485979497544,14.948597949754403 209.76341276456924,14.948597949754403 209.76341276456924,134.9485979497544 114.9485979497544,134.9485979497544" style="fill:#ECB172A0;stroke-width:0"/>

      <!-- Text -->
      <text x="162.356005" y="154.948598" font-size="1.0rem" font-weight="100" text-anchor="middle" >3072</text>
      <text x="229.763413" y="74.948598" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,229.763413,74.948598)">3888</text>
      <text x="97.474299" y="147.474299" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,97.474299,147.474299)">1</text>
    </svg>
    </td>
    </tr>
    </table></div></div></li><li class='xr-section-item'><input id='section-c871c7d6-daaa-4ad0-a7d2-20bc300cfd8e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c871c7d6-daaa-4ad0-a7d2-20bc300cfd8e' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.616e+09 1.616e+09 ... 1.616e+09</div><input id='attrs-b54aab16-6b58-48d1-859d-3d6515f45c57' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b54aab16-6b58-48d1-859d-3d6515f45c57' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7686f854-86ce-477f-81e0-054019270f82' class='xr-var-data-in' type='checkbox'><label for='data-7686f854-86ce-477f-81e0-054019270f82' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09, 1.616209e+09,
           1.616209e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09, 1.616210e+09,
           1.616210e+09])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bdb275ab-b7a8-40f3-8c03-343756be453d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bdb275ab-b7a8-40f3-8c03-343756be453d' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>object :</span></dt><dd>dexela</dd></dl></div></li></ul></div></div>



We also need the metadata of the grid scan, especially the ``shape`` of
the grid. If not provided, the calculation can still be done but the
coordinates of the grain map is unknown.

.. code:: ipython3

    # show the metadata
    metadata = dict(run.metadata["start"])
    # Because I terminate the data. I nedd to update the metadata.
    metadata["shape"] = [101]
    metadata["extents"] = ([-0.5, 0.499],)
    calculator.metadata = metadata
    calculator.metadata




.. parsed-literal::

    {'time': 1616208499.2537348,
     'uid': '257b5581-ca78-4309-9c50-b4d65d80152a',
     'versions': {'ophyd': '1.3.3', 'bluesky': '1.6.7'},
     'scan_id': 45,
     'proposal_id': '307690',
     'plan_type': 'generator',
     'plan_name': 'rel_grid_scan',
     'detectors': ['dexela'],
     'motors': ['mPhi'],
     'num_points': 1001,
     'num_intervals': 1000,
     'plan_args': {'detectors': ["XPDDDexelaDetector(prefix='XF:28IDD-ES:2{Det:DEX}', name='dexela', read_attrs=['stats1', 'stats1.total', 'tiff'], configuration_attrs=['cam', 'cam.acquire_period', 'cam.acquire_time', 'cam.image_mode', 'cam.trigger_mode', 'stats1', 'stats1.configuration_names', 'stats1.port_name', 'stats1.asyn_pipeline_config', 'stats1.blocking_callbacks', 'stats1.enable', 'stats1.nd_array_port', 'stats1.plugin_type', 'stats1.bgd_width', 'stats1.centroid_threshold', 'stats1.compute_centroid', 'stats1.compute_histogram', 'stats1.compute_profiles', 'stats1.compute_statistics', 'stats1.hist_max', 'stats1.hist_min', 'stats1.hist_size', 'stats1.profile_cursor', 'stats1.profile_size', 'stats1.ts_num_points', 'tiff', 'detector_type'])"],
      'args': ["EpicsMotor(prefix='XF:28IDD-ES:2{Stg:Stack-Ax:Phi}Mtr', name='mPhi', settle_time=0.0, timeout=None, read_attrs=['user_readback', 'user_setpoint'], configuration_attrs=['user_offset', 'user_offset_dir', 'velocity', 'acceleration', 'motor_egu'])",
       -0.5,
       0.5,
       1001],
      'per_step': 'None'},
     'hints': {'gridding': 'rectilinear', 'dimensions': [[['mPhi'], 'primary']]},
     'shape': [101],
     'extents': ([-0.5, 0.499],),
     'snaking': [False],
     'plan_pattern': 'outer_product',
     'plan_pattern_args': {'args': ["EpicsMotor(prefix='XF:28IDD-ES:2{Stg:Stack-Ax:Phi}Mtr', name='mPhi', settle_time=0.0, timeout=None, read_attrs=['user_readback', 'user_setpoint'], configuration_attrs=['user_offset', 'user_offset_dir', 'velocity', 'acceleration', 'motor_egu'])",
       -0.5,
       0.5,
       1001]},
     'plan_pattern_module': 'bluesky.plan_patterns',
     'task': 'a single point rocking curve',
     'sample': 'PARADIM-2',
     'beam': 'slit'}



We can also apply the geometry of the experiment to let the calculator
calculate the Q value of the peaks. This is optional.

.. code:: ipython3

    from  pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    calculator.ai = AzimuthalIntegrator(dist=200, wavelength=0.186, detector="dexela2923", poni1=1536, poni2=1944)

Process the data
----------------

The simplest way to use the calculator is to use the ``auto_process``.
It takes three necessary parameters. You will find the meaning of them
in the docs.

.. code:: ipython3

    help(calculator.auto_process)


.. parsed-literal::

    Help on method auto_process in module crystalmapping.utils:

    auto_process(num_wins: int, hw_wins: int, diameter: int, index_filter: slice = None, \*args, \*\*kwargs) -> None method of crystalmapping.utils.Calculator instance
        Automatically process the data in the standard protocol.

        Parameters
        ----------
        num_wins : int
            The number of windows.
        hw_wins : int
            The half width of the windows in pixels.
        diameter : int
            The diameter of the kernel to use in peak finding in pixels. It must be an odd integer.
        index_filter : slice
            The index slice of the data to use in the calculation of the dark and light image.
        args :
            The position arguments of the peak finding function `trackpy.locate`.
        kwargs :
            The keyword arguments of the peak finding function `trackpy.locate`.

        Returns
        -------
        None. The calculation results are saved in attributes.



Here we process the data. The new algorithm is a two-run-through
algorithm so there are two status bars. First one show the status of the
calculation of light and dark image and the second one shows the status
of the calculation of the crystal maps.

.. code:: ipython3

    calculator.auto_process(num_wins=4, hw_wins=25, diameter=41)


.. parsed-literal::

    100%|██████████| 100/100 [00:37<00:00,  2.65it/s]
    100%|██████████| 101/101 [00:28<00:00,  3.53it/s]


Visualize the data
------------------

All the final, intermediate and raw data can be visualized. The methods
to visualize them starts with “show”. Here, we show two examples.

Here, we show the windows on the dark subtracted light image.

.. code:: ipython3

    calculator.show_windows(vmax=500, size=8);



.. image:: _static/01_analysis_example_code_20_0.png


Then, we show the final rocking curves plot, where are the one
dimensional crystal maps.

.. code:: ipython3

    calculator.show_intensity();



.. image:: _static/01_analysis_example_code_22_0.png


Save the data
-------------

The data can be converted to ``DataSet`` and you can save it in multiple
formats.

.. code:: ipython3

    ds = calculator.to_dataset()
    ds




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */

    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }

    html[theme=dark],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1F1F1F;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }

    .xr-wrap {
      display: block;
      min-width: 300px;
      max-width: 700px;
    }

    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }

    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }

    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }

    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }

    .xr-obj-type {
      color: var(--xr-font-color2);
    }

    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }

    .xr-section-item {
      display: contents;
    }

    .xr-section-item input {
      display: none;
    }

    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }

    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }

    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }

    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }

    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }

    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }

    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }

    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }

    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }

    .xr-section-summary-in:checked + label > span {
      display: none;
    }

    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }

    .xr-section-inline-details {
      grid-column: 2 / -1;
    }

    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }

    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }

    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }

    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }

    .xr-preview {
      color: var(--xr-font-color3);
    }

    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }

    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }

    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }

    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }

    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }

    .xr-dim-list:before {
      content: '(';
    }

    .xr-dim-list:after {
      content: ')';
    }

    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }

    .xr-has-index {
      font-weight: bold;
    }

    .xr-var-list,
    .xr-var-item {
      display: contents;
    }

    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }

    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }

    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }

    .xr-var-name {
      grid-column: 1;
    }

    .xr-var-dims {
      grid-column: 2;
    }

    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }

    .xr-var-preview {
      grid-column: 4;
    }

    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }

    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }

    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }

    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }

    .xr-var-data > table {
      float: right;
    }

    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }

    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }

    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }

    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }

    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }

    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }

    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }

    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
    Dimensions:    (dim_0: 101, grain: 4, pixel_x: 3072, pixel_y: 3888)
    Coordinates:
      * grain      (grain) int64 3 2 1 0
      * dim_0      (dim_0) float64 -0.5 -0.49 -0.48 -0.47 ... 0.479 0.489 0.499
    Dimensions without coordinates: pixel_x, pixel_y
    Data variables:
        dark       (pixel_y, pixel_x) float64 300.0 303.0 300.0 ... 297.0 311.0
        light      (pixel_y, pixel_x) float64 339.0 339.0 336.0 ... 332.0 341.0
        intensity  (grain, dim_0) float64 65.69 83.61 92.62 ... 17.92 18.29 17.96
        y          (grain) int64 3809 3334 2712 1595
        dy         (grain) int64 25 25 25 25
        x          (grain) int64 200 1437 1890 109
        dx         (grain) int64 25 25 25 25
        Q          (grain) float64 4.38e-08 4.38e-08 4.38e-08 4.38e-08
    Attributes: (12/22)
        time:                 1616208499.2537348
        uid:                  257b5581-ca78-4309-9c50-b4d65d80152a
        versions:             {&#x27;ophyd&#x27;: &#x27;1.3.3&#x27;, &#x27;bluesky&#x27;: &#x27;1.6.7&#x27;}
        scan_id:              45
        proposal_id:          307690
        plan_type:            generator
        ...                   ...
        plan_pattern:         outer_product
        plan_pattern_args:    {&#x27;args&#x27;: [&quot;EpicsMotor(prefix=&#x27;XF:28IDD-ES:2{Stg:Sta...
        plan_pattern_module:  bluesky.plan_patterns
        task:                 a single point rocking curve
        sample:               PARADIM-2
        beam:                 slit</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-18d71e9f-a4df-423b-8962-c6f36cfda1e1' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-18d71e9f-a4df-423b-8962-c6f36cfda1e1' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>dim_0</span>: 101</li><li><span class='xr-has-index'>grain</span>: 4</li><li><span>pixel_x</span>: 3072</li><li><span>pixel_y</span>: 3888</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-2f83e57d-c453-4ffd-9fbb-eba9cc99ab93' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2f83e57d-c453-4ffd-9fbb-eba9cc99ab93' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>grain</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>3 2 1 0</div><input id='attrs-b195819e-fc47-4a64-ae69-c0f7828797b6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b195819e-fc47-4a64-ae69-c0f7828797b6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bde0e015-aa7f-45dc-b801-eb54ba5fd674' class='xr-var-data-in' type='checkbox'><label for='data-bde0e015-aa7f-45dc-b801-eb54ba5fd674' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([3, 2, 1, 0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>dim_0</span></div><div class='xr-var-dims'>(dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.5 -0.49 -0.48 ... 0.489 0.499</div><input id='attrs-25e2f9d9-982a-4d98-857c-3e9168f30082' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-25e2f9d9-982a-4d98-857c-3e9168f30082' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f61ade11-090e-4db3-a20e-8e0cbd17659f' class='xr-var-data-in' type='checkbox'><label for='data-f61ade11-090e-4db3-a20e-8e0cbd17659f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-5.0000e-01, -4.9001e-01, -4.8002e-01, -4.7003e-01, -4.6004e-01,
           -4.5005e-01, -4.4006e-01, -4.3007e-01, -4.2008e-01, -4.1009e-01,
           -4.0010e-01, -3.9011e-01, -3.8012e-01, -3.7013e-01, -3.6014e-01,
           -3.5015e-01, -3.4016e-01, -3.3017e-01, -3.2018e-01, -3.1019e-01,
           -3.0020e-01, -2.9021e-01, -2.8022e-01, -2.7023e-01, -2.6024e-01,
           -2.5025e-01, -2.4026e-01, -2.3027e-01, -2.2028e-01, -2.1029e-01,
           -2.0030e-01, -1.9031e-01, -1.8032e-01, -1.7033e-01, -1.6034e-01,
           -1.5035e-01, -1.4036e-01, -1.3037e-01, -1.2038e-01, -1.1039e-01,
           -1.0040e-01, -9.0410e-02, -8.0420e-02, -7.0430e-02, -6.0440e-02,
           -5.0450e-02, -4.0460e-02, -3.0470e-02, -2.0480e-02, -1.0490e-02,
           -5.0000e-04,  9.4900e-03,  1.9480e-02,  2.9470e-02,  3.9460e-02,
            4.9450e-02,  5.9440e-02,  6.9430e-02,  7.9420e-02,  8.9410e-02,
            9.9400e-02,  1.0939e-01,  1.1938e-01,  1.2937e-01,  1.3936e-01,
            1.4935e-01,  1.5934e-01,  1.6933e-01,  1.7932e-01,  1.8931e-01,
            1.9930e-01,  2.0929e-01,  2.1928e-01,  2.2927e-01,  2.3926e-01,
            2.4925e-01,  2.5924e-01,  2.6923e-01,  2.7922e-01,  2.8921e-01,
            2.9920e-01,  3.0919e-01,  3.1918e-01,  3.2917e-01,  3.3916e-01,
            3.4915e-01,  3.5914e-01,  3.6913e-01,  3.7912e-01,  3.8911e-01,
            3.9910e-01,  4.0909e-01,  4.1908e-01,  4.2907e-01,  4.3906e-01,
            4.4905e-01,  4.5904e-01,  4.6903e-01,  4.7902e-01,  4.8901e-01,
            4.9900e-01])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-df5fb8df-7bb2-40d7-9f27-24bbb8d07719' class='xr-section-summary-in' type='checkbox'  checked><label for='section-df5fb8df-7bb2-40d7-9f27-24bbb8d07719' class='xr-section-summary' >Data variables: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>dark</span></div><div class='xr-var-dims'>(pixel_y, pixel_x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>300.0 303.0 300.0 ... 297.0 311.0</div><input id='attrs-6292f8ae-b715-4d34-938b-940978f90c5f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6292f8ae-b715-4d34-938b-940978f90c5f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e17364c6-3ff4-4011-8b8f-b5086530a187' class='xr-var-data-in' type='checkbox'><label for='data-e17364c6-3ff4-4011-8b8f-b5086530a187' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[300., 303., 300., ..., 332., 334., 336.],
           [299., 306., 298., ..., 319., 327., 321.],
           [294., 310., 286., ..., 328., 321., 328.],
           ...,
           [335., 326., 330., ..., 294., 289., 308.],
           [329., 329., 322., ..., 307., 284., 310.],
           [331., 335., 320., ..., 305., 297., 311.]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>light</span></div><div class='xr-var-dims'>(pixel_y, pixel_x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>339.0 339.0 336.0 ... 332.0 341.0</div><input id='attrs-697e3faf-07f2-4937-b244-1c3d6af8f4d7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-697e3faf-07f2-4937-b244-1c3d6af8f4d7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-93ccd818-a02d-468d-8045-04c2958e1ffc' class='xr-var-data-in' type='checkbox'><label for='data-93ccd818-a02d-468d-8045-04c2958e1ffc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[339., 339., 336., ..., 365., 371., 371.],
           [331., 338., 327., ..., 349., 359., 367.],
           [330., 351., 323., ..., 358., 359., 363.],
           ...,
           [369., 371., 362., ..., 327., 329., 351.],
           [365., 371., 362., ..., 337., 321., 344.],
           [374., 373., 375., ..., 335., 332., 341.]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>intensity</span></div><div class='xr-var-dims'>(grain, dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>65.69 83.61 92.62 ... 18.29 17.96</div><input id='attrs-01d9bb7d-66b9-42cb-8757-cca3759536fc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-01d9bb7d-66b9-42cb-8757-cca3759536fc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-658b10ea-79c7-4918-b198-820824157e2e' class='xr-var-data-in' type='checkbox'><label for='data-658b10ea-79c7-4918-b198-820824157e2e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  65.68896578,   83.61399462,   92.62322184,   67.08227605,
              60.94579008,   99.78392926,  349.04344483,  767.14340638,
            1284.76970396, 1253.59400231, 1111.81545559, 1205.10111496,
            1202.78008458, 1036.17301038,  665.57362553,  438.7135717 ,
             679.0911188 ,  576.35601692,  182.34871203,   84.47327951,
              43.06305267,   31.28604383,   26.37408689,   24.64167628,
              22.94925029,   23.18762015,   22.63360246,   23.4840446 ,
              24.13533256,   24.14186851,   32.94925029,   39.15955402,
              38.71818531,   42.97270281,   77.62860438,  449.23298731,
             766.34563629,  859.0626682 ,  711.88735102,  337.99692426,
             150.49250288,  484.59515571, 1015.0615148 , 1249.66474433,
            1098.24221453,  701.61322568,  326.30757401,  310.70357555,
             133.43137255,  170.757401  ,  290.03537101,  495.9077278 ,
             746.0142253 ,  973.31026528,  578.04652057,  239.19146482,
              70.95693964,   34.69319493,   26.58246828,   24.19184929,
              22.60899654,   20.38754325,   19.6189927 ,   18.87043445,
              19.25297962,   18.46905037,   18.70319108,   18.17916186,
              18.35217224,   18.03152634,   18.03498654,   17.787005  ,
              18.8781238 ,   18.25105729,   18.22376009,   18.29950019,
              16.66051519,   17.72126105,   18.19607843,   18.58362168,
    ...
             518.91349481,  633.96693579,  701.76701269,  694.54402153,
             643.96386005,  568.69204152,  490.14455978,  395.6070742 ,
             307.99653979,  210.47635525,  140.35870819,   98.9561707 ,
              95.44675125,  120.39715494,  137.95693964,  148.76662822,
             153.4709727 ,  143.23760092,  119.76816609,   82.44290657,
              57.46482122,   35.90234525,   26.97962322,   22.91926182,
              21.73625529,   21.08150711,   20.05574779,   18.49211842,
              20.6343714 ,   18.73394848,   19.816609  ,   19.58400615,
              19.14417532,   19.26412918,   18.90811226,   18.56132257,
              18.21107266,   18.9869281 ,   18.89734717,   18.11649366,
              19.28296809,   18.41560938,   17.52210688,   18.27681661,
              18.25874664,   17.79123414,   17.75816993,   19.06113033,
              17.56978085,   18.28527489,   18.04575163,   18.59900038,
              17.58900423,   18.24221453,   18.05997693,   18.56247597,
              17.8023837 ,   18.63321799,   17.58708189,   17.77316417,
              17.40522876,   18.56978085,   17.36793541,   18.02191465,
              17.81814687,   18.16532103,   17.83660131,   18.31218762,
              19.31987697,   19.13302576,   18.09611688,   18.34717416,
              18.41830065,   18.3775471 ,   17.91926182,   18.29065744,
              17.95809304]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>3809 3334 2712 1595</div><input id='attrs-c22253d1-a61d-4003-a3a7-d8a9179f7b7f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c22253d1-a61d-4003-a3a7-d8a9179f7b7f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f7df0613-8be0-4b6e-bc3f-6400bfaf504e' class='xr-var-data-in' type='checkbox'><label for='data-f7df0613-8be0-4b6e-bc3f-6400bfaf504e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([3809, 3334, 2712, 1595])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>dy</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>25 25 25 25</div><input id='attrs-a7bf6d82-2a5e-4501-9d34-471d6306419e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a7bf6d82-2a5e-4501-9d34-471d6306419e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eaddd2f1-ef7a-464a-b92a-f2c656e0ac53' class='xr-var-data-in' type='checkbox'><label for='data-eaddd2f1-ef7a-464a-b92a-f2c656e0ac53' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([25, 25, 25, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>200 1437 1890 109</div><input id='attrs-7cd0c638-c5f5-4e88-9455-af2cd715d885' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7cd0c638-c5f5-4e88-9455-af2cd715d885' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f84a09f3-a569-4318-a0ab-8e54533eec6e' class='xr-var-data-in' type='checkbox'><label for='data-f84a09f3-a569-4318-a0ab-8e54533eec6e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 200, 1437, 1890,  109])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>dx</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>25 25 25 25</div><input id='attrs-72baf9fb-1307-4f85-836d-22ccec6b3cfe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-72baf9fb-1307-4f85-836d-22ccec6b3cfe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-37ff52a4-7651-4aab-b9a2-159d3a4bae15' class='xr-var-data-in' type='checkbox'><label for='data-37ff52a4-7651-4aab-b9a2-159d3a4bae15' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([25, 25, 25, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Q</span></div><div class='xr-var-dims'>(grain)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.38e-08 4.38e-08 4.38e-08 4.38e-08</div><input id='attrs-daf91a4a-488f-4b9c-89cd-fd774553aa83' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-daf91a4a-488f-4b9c-89cd-fd774553aa83' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3fad60f3-f87c-4f3d-bb0e-0e3ffd959b7f' class='xr-var-data-in' type='checkbox'><label for='data-3fad60f3-f87c-4f3d-bb0e-0e3ffd959b7f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([4.38002971e-08, 4.38002488e-08, 4.38002742e-08, 4.38005168e-08])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-40af546b-68bc-4213-af7c-f1f668806514' class='xr-section-summary-in' type='checkbox'  ><label for='section-40af546b-68bc-4213-af7c-f1f668806514' class='xr-section-summary' >Attributes: <span>(22)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>time :</span></dt><dd>1616208499.2537348</dd><dt><span>uid :</span></dt><dd>257b5581-ca78-4309-9c50-b4d65d80152a</dd><dt><span>versions :</span></dt><dd>{&#x27;ophyd&#x27;: &#x27;1.3.3&#x27;, &#x27;bluesky&#x27;: &#x27;1.6.7&#x27;}</dd><dt><span>scan_id :</span></dt><dd>45</dd><dt><span>proposal_id :</span></dt><dd>307690</dd><dt><span>plan_type :</span></dt><dd>generator</dd><dt><span>plan_name :</span></dt><dd>rel_grid_scan</dd><dt><span>detectors :</span></dt><dd>[&#x27;dexela&#x27;]</dd><dt><span>motors :</span></dt><dd>[&#x27;mPhi&#x27;]</dd><dt><span>num_points :</span></dt><dd>1001</dd><dt><span>num_intervals :</span></dt><dd>1000</dd><dt><span>plan_args :</span></dt><dd>{&#x27;detectors&#x27;: [&quot;XPDDDexelaDetector(prefix=&#x27;XF:28IDD-ES:2{Det:DEX}&#x27;, name=&#x27;dexela&#x27;, read_attrs=[&#x27;stats1&#x27;, &#x27;stats1.total&#x27;, &#x27;tiff&#x27;], configuration_attrs=[&#x27;cam&#x27;, &#x27;cam.acquire_period&#x27;, &#x27;cam.acquire_time&#x27;, &#x27;cam.image_mode&#x27;, &#x27;cam.trigger_mode&#x27;, &#x27;stats1&#x27;, &#x27;stats1.configuration_names&#x27;, &#x27;stats1.port_name&#x27;, &#x27;stats1.asyn_pipeline_config&#x27;, &#x27;stats1.blocking_callbacks&#x27;, &#x27;stats1.enable&#x27;, &#x27;stats1.nd_array_port&#x27;, &#x27;stats1.plugin_type&#x27;, &#x27;stats1.bgd_width&#x27;, &#x27;stats1.centroid_threshold&#x27;, &#x27;stats1.compute_centroid&#x27;, &#x27;stats1.compute_histogram&#x27;, &#x27;stats1.compute_profiles&#x27;, &#x27;stats1.compute_statistics&#x27;, &#x27;stats1.hist_max&#x27;, &#x27;stats1.hist_min&#x27;, &#x27;stats1.hist_size&#x27;, &#x27;stats1.profile_cursor&#x27;, &#x27;stats1.profile_size&#x27;, &#x27;stats1.ts_num_points&#x27;, &#x27;tiff&#x27;, &#x27;detector_type&#x27;])&quot;], &#x27;args&#x27;: [&quot;EpicsMotor(prefix=&#x27;XF:28IDD-ES:2{Stg:Stack-Ax:Phi}Mtr&#x27;, name=&#x27;mPhi&#x27;, settle_time=0.0, timeout=None, read_attrs=[&#x27;user_readback&#x27;, &#x27;user_setpoint&#x27;], configuration_attrs=[&#x27;user_offset&#x27;, &#x27;user_offset_dir&#x27;, &#x27;velocity&#x27;, &#x27;acceleration&#x27;, &#x27;motor_egu&#x27;])&quot;, -0.5, 0.5, 1001], &#x27;per_step&#x27;: &#x27;None&#x27;}</dd><dt><span>hints :</span></dt><dd>{&#x27;gridding&#x27;: &#x27;rectilinear&#x27;, &#x27;dimensions&#x27;: [[[&#x27;mPhi&#x27;], &#x27;primary&#x27;]]}</dd><dt><span>shape :</span></dt><dd>[101]</dd><dt><span>extents :</span></dt><dd>[[-0.5, 0.499]]</dd><dt><span>snaking :</span></dt><dd>[False]</dd><dt><span>plan_pattern :</span></dt><dd>outer_product</dd><dt><span>plan_pattern_args :</span></dt><dd>{&#x27;args&#x27;: [&quot;EpicsMotor(prefix=&#x27;XF:28IDD-ES:2{Stg:Stack-Ax:Phi}Mtr&#x27;, name=&#x27;mPhi&#x27;, settle_time=0.0, timeout=None, read_attrs=[&#x27;user_readback&#x27;, &#x27;user_setpoint&#x27;], configuration_attrs=[&#x27;user_offset&#x27;, &#x27;user_offset_dir&#x27;, &#x27;velocity&#x27;, &#x27;acceleration&#x27;, &#x27;motor_egu&#x27;])&quot;, -0.5, 0.5, 1001]}</dd><dt><span>plan_pattern_module :</span></dt><dd>bluesky.plan_patterns</dd><dt><span>task :</span></dt><dd>a single point rocking curve</dd><dt><span>sample :</span></dt><dd>PARADIM-2</dd><dt><span>beam :</span></dt><dd>slit</dd></dl></div></li></ul></div></div>



Here, we save it in ``NetCDF`` format. Before it is saved, the ``attrs``
need to be cleaned.

.. code:: ipython3

    ds.attrs = {}
    ds.to_netcdf("data/example.nc")

Load the dataset and visualize it
---------------------------------

The data can be loaded and visualized again after the data processing
session is over.

.. code:: ipython3

    import xarray as xr

    ds = xr.load_dataset("data/example.nc")

.. code:: ipython3

    calculator.auto_visualize(ds);



.. image:: _static/01_analysis_example_code_29_0.png

