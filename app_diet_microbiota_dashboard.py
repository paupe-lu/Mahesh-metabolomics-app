from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


APP_TITLE = "Diet x Microbiota Metabolomics Explorer"
UPDATED_WORKBOOK = Path(__file__).with_name("Mahesh_metabolomics_updated.xlsx")
LEGACY_WORKBOOK = Path(__file__).with_name("Mahesh_metabolomics.xlsx")
PSEUDOCOUNT = 1e-9
SIGNIFICANCE_Q = 0.05
LOG2FC_THRESHOLD = 1
GF_14SM_GROUPS_ORDER = ["GF_SC1", "GF_FF", "14SM_SC1", "14SM_FF"]
SPF_GROUPS_ORDER = ["SPF_SC1", "SPF_SC2", "SPF_IN", "SPF_FS", "SPF_FF"]
GF_14SM_COLONIZATION_ORDER = ["GF", "14SM"]
SPF_COLONIZATION_ORDER = ["SPF"]
GF_14SM_DIET_ORDER = ["SC1", "FF"]
SPF_DIET_ORDER = ["SC1", "SC2", "IN", "FS", "FF"]
GF_14SM_COMPARISONS = {
    "GF: FF vs SC1": ("GF_FF", "GF_SC1"),
    "14SM: FF vs SC1": ("14SM_FF", "14SM_SC1"),
    "SC1: 14SM vs GF": ("14SM_SC1", "GF_SC1"),
    "FF: 14SM vs GF": ("14SM_FF", "GF_FF"),
}


def benjamini_hochberg(p_values):
    p = pd.Series(p_values, dtype=float)
    valid = p.notna()
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)

    if valid.sum() == 0:
        return adjusted

    ranked = p[valid].sort_values()
    n = len(ranked)
    raw_q = ranked * n / np.arange(1, n + 1)
    adjusted_sorted = raw_q.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    adjusted.loc[adjusted_sorted.index] = adjusted_sorted
    return adjusted.reindex(p.index)


def group_values(row, metadata, group):
    samples = metadata.loc[metadata.Group == group, "Sample"]
    return row[samples].dropna().astype(float)


def mean_delta(row, metadata, group_b, group_a):
    b = group_values(row, metadata, group_b)
    a = group_values(row, metadata, group_a)
    return b.mean() - a.mean()


def log2_fold_change(row, metadata, group_b, group_a):
    b = group_values(row, metadata, group_b)
    a = group_values(row, metadata, group_a)
    return np.log2((b.mean() + PSEUDOCOUNT) / (a.mean() + PSEUDOCOUNT))


def welch_p_value(row, metadata, group_b, group_a):
    b = group_values(row, metadata, group_b)
    a = group_values(row, metadata, group_a)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    return ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue


def comparison_table(metabolites, metadata, group_b, group_a):
    rows = []

    for _, row in metabolites.iterrows():
        p = welch_p_value(row, metadata, group_b, group_a)
        rows.append(
            [
                row["KEGG"],
                row["Metabolite"],
                log2_fold_change(row, metadata, group_b, group_a),
                mean_delta(row, metadata, group_b, group_a),
                p,
            ]
        )

    result = pd.DataFrame(
        rows,
        columns=["KEGG", "Metabolite", "log2FC", "mean_delta", "p"],
    )
    result["q"] = benjamini_hochberg(result["p"])
    result["minuslog10p"] = -np.log10(result["p"] + 1e-300)
    return result


def classify_diet_response(row):
    gf_sig = (
        pd.notna(row["GF_q"])
        and row["GF_q"] < SIGNIFICANCE_Q
        and abs(row["GF_log2FC"]) >= LOG2FC_THRESHOLD
    )
    sm_sig = (
        pd.notna(row["14SM_q"])
        and row["14SM_q"] < SIGNIFICANCE_Q
        and abs(row["14SM_log2FC"]) >= LOG2FC_THRESHOLD
    )

    if gf_sig and sm_sig:
        if np.sign(row["GF_log2FC"]) == np.sign(row["14SM_log2FC"]):
            return "Both same direction"
        return "Both opposite direction"

    if gf_sig:
        return "GF only"

    if sm_sig:
        return "14SM only"

    return "Not significant"


def default_workbook():
    if UPDATED_WORKBOOK.exists():
        return UPDATED_WORKBOOK
    if LEGACY_WORKBOOK.exists():
        return LEGACY_WORKBOOK
    return None


def workbook_label(workbook):
    if isinstance(workbook, Path):
        return workbook.name
    return getattr(workbook, "name", "uploaded workbook")


def clean_sample_name(value):
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def is_outlier_sample(sample_name):
    return "outlier" in str(sample_name).lower()


def detect_unit(raw):
    first_cell = str(raw.iloc[0, 0])
    if "nmol/g" in first_cell:
        return "nmol/g"
    if "nmol/kg" in first_cell:
        return "nmol/kg"
    return "nmol/g"


def build_gf_14sm_metadata(raw):
    metadata = []

    for col in range(2, 18):
        sample = clean_sample_name(raw.iloc[4, col])
        colonization = "14SM" if col <= 9 else "GF"
        diet = "SC1" if col in [2, 3, 4, 5, 10, 11, 12, 13] else "FF"

        metadata.append(
            {
                "Sample": sample,
                "Colonization": colonization,
                "Diet": diet,
                "Group": f"{colonization}_{diet}",
                "Experiment": "GF / 14SM",
            }
        )

    return pd.DataFrame(metadata)


def build_spf_metadata(raw):
    metadata = []
    diet_by_col = {
        **dict.fromkeys(range(2, 6), "SC1"),
        **dict.fromkeys(range(6, 10), "SC2"),
        **dict.fromkeys(range(10, 14), "IN"),
        **dict.fromkeys(range(14, 18), "FS"),
        **dict.fromkeys(range(18, 22), "FF"),
    }

    for col in range(2, 22):
        sample = clean_sample_name(raw.iloc[3, col])
        if is_outlier_sample(sample):
            continue

        diet = diet_by_col[col]

        metadata.append(
            {
                "Sample": sample,
                "Colonization": "SPF",
                "Diet": diet,
                "Group": f"SPF_{diet}",
                "Experiment": "SPF diet",
            }
        )

    return pd.DataFrame(metadata)


def build_metabolite_table(raw, sample_names, start_row, sample_cols):
    metabolites = raw.iloc[start_row:, [0, 1] + sample_cols].copy()
    metabolites.columns = ["KEGG", "Metabolite"] + sample_names
    metabolites = metabolites.dropna(subset=["Metabolite"])

    for sample in sample_names:
        metabolites[sample] = pd.to_numeric(metabolites[sample], errors="coerce")

    return metabolites


def load_experiment(workbook, experiment_label):
    if hasattr(workbook, "seek"):
        workbook.seek(0)
    excel = pd.ExcelFile(workbook)
    sheet_names = excel.sheet_names

    if experiment_label == "SPF diet":
        if hasattr(workbook, "seek"):
            workbook.seek(0)
        raw = pd.read_excel(workbook, sheet_name="SPF_diet", header=None)
        sample_cols = [
            col
            for col in range(2, 22)
            if not is_outlier_sample(clean_sample_name(raw.iloc[3, col]))
        ]
        sample_names = [
            clean_sample_name(raw.iloc[3, col])
            for col in sample_cols
        ]
        metadata = build_spf_metadata(raw)
        metabolites = build_metabolite_table(raw, sample_names, start_row=5, sample_cols=sample_cols)
        return raw, metadata, metabolites, sample_names, detect_unit(raw)

    sheet_name = "GF_14SM" if "GF_14SM" in sheet_names else 0
    if hasattr(workbook, "seek"):
        workbook.seek(0)
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    sample_cols = list(range(2, 18))
    sample_names = [clean_sample_name(raw.iloc[4, col]) for col in sample_cols]
    metadata = build_gf_14sm_metadata(raw)
    metabolites = build_metabolite_table(raw, sample_names, start_row=6, sample_cols=sample_cols)
    return raw, metadata, metabolites, sample_names, detect_unit(raw)


def available_experiments(workbook):
    if hasattr(workbook, "seek"):
        workbook.seek(0)
    sheet_names = pd.ExcelFile(workbook).sheet_names
    experiments = ["GF / 14SM"]
    if "SPF_diet" in sheet_names:
        experiments.append("SPF diet")
    return experiments


def filter_metadata(metadata, colonization, diet):
    filtered = metadata.copy()

    if colonization != "All":
        filtered = filtered[filtered["Colonization"] == colonization]

    if diet != "All":
        filtered = filtered[filtered["Diet"] == diet]

    return filtered


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(APP_TITLE)

st.sidebar.title("Data")
st.sidebar.caption("A bundled workbook is loaded automatically. Upload another Excel file to override it.")
uploaded = st.sidebar.file_uploader(
    "Workbook",
    type=["xlsx"],
    help="Upload a Mahesh metabolomics .xlsx workbook to replace the bundled file.",
)

if uploaded is not None:
    workbook = uploaded
    st.sidebar.success(f"Using uploaded workbook: {workbook_label(workbook)}")
else:
    workbook = default_workbook()

if workbook is None:
    st.info("Upload the Mahesh metabolomics Excel file.")
    st.stop()
elif uploaded is None:
    st.sidebar.info(f"Auto-loaded workbook: {workbook_label(workbook)}")

try:
    experiments = available_experiments(workbook)
except Exception as exc:
    st.error(f"Could not read workbook '{workbook_label(workbook)}'.")
    st.exception(exc)
    st.stop()

selected_experiment = st.sidebar.selectbox("Experiment", experiments)

try:
    raw, meta, met, sample_names, unit = load_experiment(workbook, selected_experiment)
except Exception as exc:
    st.error(f"Could not load the '{selected_experiment}' experiment from '{workbook_label(workbook)}'.")
    st.exception(exc)
    st.stop()
groups_order = GF_14SM_GROUPS_ORDER if selected_experiment == "GF / 14SM" else SPF_GROUPS_ORDER
colonization_order = (
    GF_14SM_COLONIZATION_ORDER
    if selected_experiment == "GF / 14SM"
    else SPF_COLONIZATION_ORDER
)
diet_order = GF_14SM_DIET_ORDER if selected_experiment == "GF / 14SM" else SPF_DIET_ORDER

metabolites = sorted(met["Metabolite"].astype(str).unique())

st.sidebar.header("Sample Filter")
selected_colonization = st.sidebar.selectbox(
    "Colonization",
    ["All"] + colonization_order,
)
selected_diet = st.sidebar.selectbox(
    "Diet",
    ["All"] + diet_order,
)

filtered_meta = filter_metadata(meta, selected_colonization, selected_diet)

st.sidebar.header("Metabolite Explorer")
selected_metabolite = st.sidebar.selectbox("Metabolite", metabolites)

st.sidebar.header("Heatmap")
use_all_heatmap_metabolites = st.sidebar.checkbox("Use all metabolites", value=False)
heatmap_top_n = st.sidebar.slider(
    "Heatmap metabolites",
    min_value=10,
    max_value=min(150, len(met)),
    value=min(50, len(met)),
    step=10,
    disabled=use_all_heatmap_metabolites,
)
heatmap_log_transform = st.sidebar.checkbox(f"Heatmap log10({unit} + 1)", value=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "PCA",
        "Metabolite Explorer",
        "Heatmap",
        "Volcano",
        "Microbiota Effect",
        "Diet Effect",
    ]
)

with tab1:
    st.markdown(
        f"""
        PCA summarizes the overall metabolomics profile of each sample in two dimensions.
        Intensities are first transformed as `log10({unit} + 1)`, missing values are filled
        with `0`, and each metabolite is standardized before PCA. Points that sit close
        together have more similar global metabolite profiles.
        """
    )

    X = met[sample_names].T.fillna(0)
    Xs = StandardScaler().fit_transform(np.log10(X + 1))

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)

    pca_df = meta.copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Colonization",
        symbol="Diet",
        hover_name="Sample",
    )
    fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")

    st.plotly_chart(fig, width="stretch")

with tab2:
    if filtered_meta.empty:
        st.warning("No samples match the selected colonization and diet filters.")
    else:
        row = met.loc[met["Metabolite"].astype(str) == selected_metabolite].iloc[0]

        plot_df = filtered_meta.copy()
        plot_df[unit] = [row[sample] for sample in plot_df["Sample"]]

        fig = px.box(
            plot_df,
            x="Group",
            y=unit,
            color="Group",
            category_orders={"Group": groups_order},
            points="all",
        )
        fig.update_traces(pointpos=0, jitter=0.25)
        fig.update_layout(yaxis_title=unit)

        st.plotly_chart(fig, width="stretch")

with tab3:
    if filtered_meta.empty:
        st.warning("No samples match the selected colonization and diet filters.")
    else:
        heat_samples = filtered_meta["Sample"].tolist()
        heat_matrix = met.set_index("Metabolite")[heat_samples].copy()

        if heatmap_log_transform:
            heat_matrix = np.log10(heat_matrix + 1)

        if len(heat_samples) < 2:
            st.warning("Select at least two samples for the heatmap.")
        elif heat_matrix.dropna(how="all").empty:
            st.warning("No metabolite values are available for the selected samples.")
        elif use_all_heatmap_metabolites:
            displayed_metabolites = heat_matrix.index
            heatmap_title = "All metabolites"
        else:
            variances = heat_matrix.var(axis=1).sort_values(ascending=False)
            displayed_metabolites = variances.head(heatmap_top_n).index
            heatmap_title = f"Top {len(displayed_metabolites)} variable metabolites"

        if len(heat_samples) >= 2 and not heat_matrix.dropna(how="all").empty:
            heat = heat_matrix.loc[displayed_metabolites]
            heat_z = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1), axis=0)
            heat_z = heat_z.replace([np.inf, -np.inf], np.nan).fillna(0)
            color_limit = float(np.nanmax(np.abs(heat_z.to_numpy())))
            if color_limit == 0 or np.isnan(color_limit):
                color_limit = 1.0

            sample_labels = filtered_meta.set_index("Sample").loc[heat_samples].apply(
                lambda r: f"{r.name} | {r['Group']}",
                axis=1,
            )
            heat_z.columns = sample_labels.tolist()

            fig = px.imshow(
                heat_z,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-color_limit,
                zmax=color_limit,
                color_continuous_midpoint=0,
                labels={"x": "Sample", "y": "Metabolite", "color": "Row z-score"},
                title=heatmap_title,
            )
            fig.update_layout(height=850)

            st.plotly_chart(fig, width="stretch")
            st.dataframe(heat_z, width="stretch")

with tab4:
    if selected_experiment == "GF / 14SM":
        comparisons = GF_14SM_COMPARISONS
    else:
        comparisons = {
            "SPF: SC2 vs SC1": ("SPF_SC2", "SPF_SC1"),
            "SPF: IN vs SC1": ("SPF_IN", "SPF_SC1"),
            "SPF: FS vs SC1": ("SPF_FS", "SPF_SC1"),
            "SPF: FF vs SC1": ("SPF_FF", "SPF_SC1"),
            "SPF: FF vs SC2": ("SPF_FF", "SPF_SC2"),
            "SPF: FF vs IN": ("SPF_FF", "SPF_IN"),
            "SPF: FF vs FS": ("SPF_FF", "SPF_FS"),
        }

    st.markdown(
        """
        The volcano plot compares two selected groups metabolite by metabolite. The x-axis is
        `log2FC`, calculated as `log2(mean comparison group / mean reference group)` with a
        small pseudocount. The y-axis is `-log10(p)`, where p-values come from Welch's t-test
        after dropping missing values. The `q` column is Benjamini-Hochberg FDR correction.
        Highlighted points have `q < 0.05` and absolute `log2FC >= 1`.
        """
    )

    comparison_label = st.selectbox("Comparison", list(comparisons.keys()))
    group_b, group_a = comparisons[comparison_label]

    vol = comparison_table(met, meta, group_b, group_a)
    vol["Significant"] = (
        (vol["q"] < SIGNIFICANCE_Q)
        & (vol["log2FC"].abs() >= LOG2FC_THRESHOLD)
    )

    fig = px.scatter(
        vol,
        x="log2FC",
        y="minuslog10p",
        color="Significant",
        hover_name="Metabolite",
        hover_data=["KEGG", "mean_delta", "p", "q"],
    )
    fig.add_vline(x=-LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
    fig.add_vline(x=LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
    fig.add_hline(y=-np.log10(SIGNIFICANCE_Q), line_dash="dash", line_color="gray")

    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        vol.sort_values(["q", "p"], na_position="last"),
        width="stretch",
    )
    st.download_button(
        "Download volcano results",
        vol.to_csv(index=False).encode("utf-8"),
        file_name=f"volcano_{group_b}_vs_{group_a}.csv",
        mime="text/csv",
    )

with tab5:
    if selected_experiment != "GF / 14SM":
        st.info("Microbiota effect is only defined for the GF / 14SM experiment.")
    else:
        st.markdown(
            f"""
            Microbiota effect estimates how colonization changes each metabolite within each diet.
            For the SC1 diet, the comparison is `14SM_SC1` versus `GF_SC1`; for the FF diet, it is
            `14SM_FF` versus `GF_FF`. Positive log2FC values mean the metabolite is higher in
            14SM-colonized mice than in germ-free mice for that diet. Mean deltas are simple
            differences in `{unit}`; p-values use Welch's t-test and q-values use FDR correction.
            """
        )

        sc_effect = comparison_table(met, meta, "14SM_SC1", "GF_SC1").rename(
            columns={
                "log2FC": "SC1_log2FC",
                "mean_delta": "SC1_mean_delta",
                "p": "SC1_p",
                "q": "SC1_q",
            }
        )
        ff_effect = comparison_table(met, meta, "14SM_FF", "GF_FF").rename(
            columns={
                "log2FC": "FF_log2FC",
                "mean_delta": "FF_mean_delta",
                "p": "FF_p",
                "q": "FF_q",
            }
        )
        eff = sc_effect[
            ["KEGG", "Metabolite", "SC1_log2FC", "SC1_mean_delta", "SC1_p", "SC1_q"]
        ].merge(
            ff_effect[["KEGG", "Metabolite", "FF_log2FC", "FF_mean_delta", "FF_p", "FF_q"]],
            on=["KEGG", "Metabolite"],
            how="left",
        )

        st.dataframe(
            eff.sort_values("SC1_log2FC", ascending=False),
            width="stretch",
        )
        st.download_button(
            "Download microbiota effects",
            eff.to_csv(index=False).encode("utf-8"),
            file_name="microbiota_effects.csv",
            mime="text/csv",
        )

with tab6:
    if selected_experiment == "GF / 14SM":
        st.markdown(
            f"""
            Diet effect estimates how the FF diet changes each metabolite relative to SC1 within
            each colonization state. For germ-free mice, the comparison is `GF_FF` versus `GF_SC1`;
            for 14SM-colonized mice, it is `14SM_FF` versus `14SM_SC1`. Positive log2FC values mean
            the metabolite is higher on FF than SC1. Mean deltas are differences in `{unit}`;
            p-values use Welch's t-test and q-values use FDR correction. Metabolites are classified
            as diet-responsive in GF only, 14SM only, both colonization states in the same direction,
            or both in opposite directions using `q < 0.05` and absolute `log2FC >= 1`.
            """
        )

        gf_effect = comparison_table(met, meta, "GF_FF", "GF_SC1").rename(
            columns={
                "log2FC": "GF_log2FC",
                "mean_delta": "GF_mean_delta",
                "p": "GF_p",
                "q": "GF_q",
            }
        )
        sm_effect = comparison_table(met, meta, "14SM_FF", "14SM_SC1").rename(
            columns={
                "log2FC": "14SM_log2FC",
                "mean_delta": "14SM_mean_delta",
                "p": "14SM_p",
                "q": "14SM_q",
            }
        )
        eff = gf_effect[
            ["KEGG", "Metabolite", "GF_log2FC", "GF_mean_delta", "GF_p", "GF_q"]
        ].merge(
            sm_effect[
                ["KEGG", "Metabolite", "14SM_log2FC", "14SM_mean_delta", "14SM_p", "14SM_q"]
            ],
            on=["KEGG", "Metabolite"],
            how="left",
        )
        eff["Diet_response"] = eff.apply(classify_diet_response, axis=1)

        response_order = [
            "GF only",
            "14SM only",
            "Both same direction",
            "Both opposite direction",
            "Not significant",
        ]

        fig = px.scatter(
            eff,
            x="GF_log2FC",
            y="14SM_log2FC",
            color="Diet_response",
            category_orders={"Diet_response": response_order},
            hover_name="Metabolite",
            hover_data=["KEGG", "GF_q", "14SM_q", "GF_mean_delta", "14SM_mean_delta"],
            labels={
                "GF_log2FC": "GF diet effect, log2FC (FF vs SC1)",
                "14SM_log2FC": "14SM diet effect, log2FC (FF vs SC1)",
                "Diet_response": "Diet response",
            },
        )
        fig.add_vline(x=0, line_color="gray")
        fig.add_hline(y=0, line_color="gray")
        fig.add_vline(x=-LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
        fig.add_vline(x=LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
        fig.add_hline(y=-LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
        fig.add_hline(y=LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")

        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            eff["Diet_response"].value_counts().reindex(response_order, fill_value=0)
            .rename_axis("Diet response")
            .reset_index(name="Metabolites"),
            width="stretch",
        )

        st.dataframe(
            eff.sort_values(["Diet_response", "GF_q", "14SM_q"], na_position="last"),
            width="stretch",
        )
        st.download_button(
            "Download diet effects",
            eff.to_csv(index=False).encode("utf-8"),
            file_name="diet_effects_gf_14sm.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            f"""
            SPF diet effect compares two diets within SPF mice. Positive log2FC values mean the
            metabolite is higher in the comparison diet than in the reference diet. Mean deltas
            are differences in `{unit}`; p-values use Welch's t-test and q-values use FDR
            correction. Highlighted metabolites use `q < 0.05` and absolute `log2FC >= 1`.
            """
        )

        col_ref, col_comp = st.columns(2)
        reference_diet = col_ref.selectbox("Reference diet", SPF_DIET_ORDER, index=0)
        comparison_diet = col_comp.selectbox("Comparison diet", SPF_DIET_ORDER, index=4)

        if reference_diet == comparison_diet:
            st.warning("Choose two different diets for the SPF diet effect.")
        else:
            group_a = f"SPF_{reference_diet}"
            group_b = f"SPF_{comparison_diet}"
            eff = comparison_table(met, meta, group_b, group_a)
            eff["Significant"] = (
                (eff["q"] < SIGNIFICANCE_Q)
                & (eff["log2FC"].abs() >= LOG2FC_THRESHOLD)
            )

            fig = px.scatter(
                eff,
                x="log2FC",
                y="minuslog10p",
                color="Significant",
                hover_name="Metabolite",
                hover_data=["KEGG", "mean_delta", "p", "q"],
                labels={
                    "log2FC": f"SPF diet effect, log2FC ({comparison_diet} vs {reference_diet})",
                    "minuslog10p": "-log10(p)",
                },
            )
            fig.add_vline(x=-LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
            fig.add_vline(x=LOG2FC_THRESHOLD, line_dash="dash", line_color="gray")
            fig.add_hline(y=-np.log10(SIGNIFICANCE_Q), line_dash="dash", line_color="gray")

            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                eff.sort_values(["q", "p"], na_position="last"),
                width="stretch",
            )
            st.download_button(
                "Download SPF diet effects",
                eff.to_csv(index=False).encode("utf-8"),
                file_name=f"diet_effects_spf_{comparison_diet}_vs_{reference_diet}.csv",
                mime="text/csv",
            )
