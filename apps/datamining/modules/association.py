"""Modul Association Rule Mining (Apriori / FP-Growth)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from core import algviz as AV
from core import data_io as IO
from core import params as PR
from core import paramui as PUI
from core import settings as SET
from core import ui as U
from core import viz as V
from core.i18n import L

FORMATS = {
    "long": ("Format panjang (kolom ID transaksi + kolom item)",
             "Long format (transaction ID column + item column)"),
    "basket": ("Format keranjang (satu baris = satu transaksi, item sebagai kolom 0/1)",
               "Basket format (one row = one transaction, items as 0/1 columns)"),
    "delimited": ("Satu kolom berisi daftar item dipisah tanda baca (mis. 'roti, susu, telur')",
                  "One column holding a delimited item list (e.g. 'bread, milk, eggs')"),
}


def _guess_format(df: pd.DataFrame) -> str:
    num = df.select_dtypes(include=[np.number])
    binary_like = [c for c in num.columns if set(pd.unique(num[c].dropna())) <= {0, 1}]
    if len(binary_like) >= max(3, df.shape[1] - 2):
        return "basket"
    obj = [c for c in df.columns if IO.is_text_dtype(df[c])]
    for c in obj:
        if df[c].astype(str).str.contains(r"[,;|]").mean() > 0.5:
            return "delimited"
    return "long"


def config(df: pd.DataFrame) -> dict | None:
    SET.consume_pending("association")
    st.markdown("### " + L("Konfigurasi Association Rule Mining", "Association Rule Mining Setup"))
    guess = _guess_format(df)
    keys = list(FORMATS)
    fmt = SET.radio(L("Format data transaksi", "Transaction data format"), keys,
                    key="ar_fmt", index=keys.index(guess),
                    format_func=lambda k: L(*FORMATS[k]))
    st.caption(L(f"Format terdeteksi otomatis: **{L(*FORMATS[guess])}**",
                 f"Auto-detected format: **{L(*FORMATS[guess])}**"))

    params = {"fmt": fmt}
    if fmt == "long":
        c1, c2 = st.columns(2)
        cand_id = [c for c in df.columns if df[c].nunique() < len(df)] or list(df.columns)
        params["id_col"] = SET.selectbox(L("Kolom ID transaksi", "Transaction ID column"),
                                         list(df.columns), key="ar_id",
                                         index=list(df.columns).index(cand_id[0]), container=c1)
        rest = [c for c in df.columns if c != params["id_col"]]
        params["item_col"] = SET.selectbox(L("Kolom item", "Item column"), rest, key="ar_item",
                                           index=0, container=c2)
    elif fmt == "basket":
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and set(pd.unique(df[c].dropna())) <= {0, 1}]
        params["item_cols"] = SET.multiselect(L("Kolom item (biner 0/1)", "Item columns (binary 0/1)"),
                                              list(df.columns), key="ar_cols", default=cols)
    else:
        c1, c2 = st.columns(2)
        params["list_col"] = SET.selectbox(L("Kolom daftar item", "Item list column"),
                                           list(df.columns), key="ar_list", container=c1)
        params["sep"] = SET.text_input(L("Pemisah", "Separator"), ",", key="ar_sep", container=c2)

    algo = SET.selectbox(L("Algoritma penambangan", "Mining algorithm"), PR.algos("association"),
                         key="ar_algo",
                         help=L("Apriori memindai data berulang kali per tingkat; FP-Growth "
                                "membangun pohon terkompresi sekali jalan sehingga lebih cepat "
                                "pada data besar. Keduanya menghasilkan itemset yang sama.",
                                "Apriori scans the data once per level; FP-Growth builds a "
                                "compressed tree in one pass and is faster on large data. Both "
                                "produce the same itemsets."))
    aparams = PUI.editor("association", [algo], "ar", expanded=True,
                         title=L(f"Parameter algoritma — {algo}", f"Algorithm parameters — {algo}"))
    rp = PR.resolve("association", algo, aparams.get(algo, {}))
    params.update(algo=algo, params=aparams,
                  min_sup=float(rp["min_support"]),
                  metric=str(rp["metric"]),
                  min_conf=float(rp["min_threshold"]),
                  min_lift=float(rp["min_lift"]),
                  max_len=int(rp["max_len"]),
                  low_memory=bool(rp.get("low_memory", False)))
    SET.panel("association", params)
    return params


def _to_basket(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["fmt"] == "long":
        d = df[[cfg["id_col"], cfg["item_col"]]].dropna().astype(str)
        d["_v"] = 1
        basket = d.pivot_table(index=cfg["id_col"], columns=cfg["item_col"],
                               values="_v", aggfunc="max", fill_value=0)
    elif cfg["fmt"] == "basket":
        basket = df[cfg["item_cols"]].fillna(0)
        basket = (basket.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(int)
    else:
        sep = cfg.get("sep") or ","
        rows = df[cfg["list_col"]].dropna().astype(str)
        recs = [[i.strip() for i in r.split(sep) if i.strip()] for r in rows]
        items = sorted({i for r in recs for i in r})
        basket = pd.DataFrame(
            [[1 if it in set(r) else 0 for it in items] for r in recs],
            columns=items, index=[f"T{i + 1}" for i in range(len(recs))])
    basket.columns = [str(c) for c in basket.columns]
    return (basket > 0).astype(bool)


def _fmt_set(s) -> str:
    return ", ".join(sorted(map(str, s)))


def run(df: pd.DataFrame, cfg: dict):
    tables, figs = {}, {}
    try:
        basket = _to_basket(df, cfg)
    except Exception as e:
        st.error(L(f"Gagal mengubah data ke format keranjang: {e}",
                   f"Failed to convert data to basket format: {e}"))
        return
    if basket.shape[1] < 2 or len(basket) < 2:
        st.error(L("Data terlalu sedikit: butuh minimal 2 transaksi dan 2 item.",
                   "Not enough data: at least 2 transactions and 2 items are required."))
        return

    sizes = basket.sum(axis=1)
    item_counts = basket.sum(axis=0).sort_values(ascending=False)
    item_support = (item_counts / len(basket)).round(4)

    # ------------------------------------------------- statistik transaksi
    U.section(L("1. Statistik Data Transaksi", "1. Transaction Data Statistics"))
    U.metric_row([
        (L("Jumlah transaksi", "Transactions"), f"{len(basket):,}"),
        (L("Jumlah item unik", "Unique items"), f"{basket.shape[1]:,}"),
        (L("Rata-rata item/transaksi", "Avg items/transaction"), f"{sizes.mean():.2f}"),
        (L("Kepadatan matriks (%)", "Matrix density (%)"), f"{basket.values.mean() * 100:.2f}"),
    ])
    ringkas = pd.DataFrame([
        (L("Jumlah transaksi", "Number of transactions"), len(basket)),
        (L("Jumlah item unik", "Number of unique items"), basket.shape[1]),
        (L("Total item terjual (kemunculan)", "Total item occurrences"), int(basket.values.sum())),
        (L("Rata-rata item per transaksi", "Mean items per transaction"), round(float(sizes.mean()), 4)),
        (L("Median item per transaksi", "Median items per transaction"), float(sizes.median())),
        (L("Std item per transaksi", "Std items per transaction"), round(float(sizes.std()), 4)),
        (L("Minimum item per transaksi", "Min items per transaction"), int(sizes.min())),
        (L("Maksimum item per transaksi", "Max items per transaction"), int(sizes.max())),
        (L("Kepadatan matriks (%)", "Matrix density (%)"), round(float(basket.values.mean() * 100), 4)),
    ], columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
    tables[L("Ringkasan Transaksi", "Transaction Summary")] = ringkas

    item_df = pd.DataFrame({
        L("Item", "Item"): item_counts.index,
        L("Frekuensi", "Frequency"): item_counts.values,
        L("Support", "Support"): item_support.values,
        L("Support (%)", "Support (%)"): (item_support.values * 100).round(2),
    })
    tables[L("Statistik Item", "Item Statistics")] = item_df

    s1, s2 = st.columns([1, 1])
    with s1:
        U.show_table(ringkas)
    with s2:
        U.show_table(item_df.head(20))

    figs[L("Frekuensi item", "Item frequency")] = V.fig_item_frequency(item_counts)
    figs[L("Ukuran keranjang", "Basket size")] = V.fig_basket_size(sizes)
    b1, b2 = st.columns(2)
    b1.pyplot(figs[L("Frekuensi item", "Item frequency")], width="stretch")
    b2.pyplot(figs[L("Ukuran keranjang", "Basket size")], width="stretch")

    # -------------------------------------------------------- itemset sering
    U.section(L("2. Itemset Sering (Frequent Itemsets)", "2. Frequent Itemsets"))
    fn = apriori if cfg["algo"] == "Apriori" else fpgrowth
    with st.spinner(L("Menambang itemset...", "Mining itemsets...")):
        kw = dict(min_support=cfg["min_sup"], use_colnames=True, max_len=cfg["max_len"])
        if cfg["algo"] == "Apriori" and cfg.get("low_memory"):
            kw["low_memory"] = True
        freq = fn(basket, **kw)
    ptab = PUI.combined_params_table("association", [cfg["algo"]], cfg.get("params", {}))
    if not ptab.empty:
        tables[L("Parameter Algoritma", "Algorithm Parameters")] = ptab
    if freq.empty:
        st.error(L(f"Tidak ada itemset dengan support ≥ {cfg['min_sup']}. Turunkan nilai min. support.",
                   f"No itemsets with support ≥ {cfg['min_sup']}. Lower the min. support."))
        return
    freq = freq.copy()
    freq["length"] = freq["itemsets"].apply(len)
    freq_view = pd.DataFrame({
        L("Itemset", "Itemset"): freq["itemsets"].apply(_fmt_set),
        L("Jumlah Item", "Itemset Size"): freq["length"],
        "Support": freq["support"].round(4),
        L("Jumlah Transaksi", "Transaction Count"): (freq["support"] * len(basket)).round(0).astype(int),
    }).sort_values("Support", ascending=False).reset_index(drop=True)
    tables[L("Itemset Sering", "Frequent Itemsets")] = freq_view
    dist = freq.groupby("length").size().reset_index(name=L("Jumlah Itemset", "Itemset Count"))
    dist.columns = [L("Panjang Itemset", "Itemset Length"), L("Jumlah Itemset", "Itemset Count")]
    tables[L("Distribusi Panjang Itemset", "Itemset Length Distribution")] = dist
    U.metric_row([
        (L("Total itemset sering", "Total frequent itemsets"), f"{len(freq):,}"),
        (L("Itemset 1-item", "1-itemsets"), f"{int((freq['length'] == 1).sum()):,}"),
        (L("Itemset 2-item", "2-itemsets"), f"{int((freq['length'] == 2).sum()):,}"),
        (L("Itemset ≥3 item", "3+-itemsets"), f"{int((freq['length'] >= 3).sum()):,}"),
    ])
    U.show_table(freq_view.head(50), height=380)

    # ------------------------------------------------------------- aturan
    U.section(L("3. Aturan Asosiasi", "3. Association Rules"))
    try:
        rules = association_rules(freq, metric=cfg.get("metric", "confidence"),
                                  min_threshold=cfg["min_conf"])
    except TypeError:
        rules = association_rules(freq, metric=cfg.get("metric", "confidence"),
                                  min_threshold=cfg["min_conf"], num_itemsets=len(freq))
    rules = rules[rules["lift"] >= cfg["min_lift"]].copy()
    if rules.empty:
        st.error(L("Tidak ada aturan yang memenuhi ambang confidence & lift. Turunkan ambangnya.",
                   "No rules meet the confidence & lift thresholds. Lower them."))
        U.export_bar(tables, figs, "asosiasi", {L("Proses", "Process"): "Association Rule Mining"})
        return

    rules["antecedents_str"] = rules["antecedents"].apply(_fmt_set)
    rules["consequents_str"] = rules["consequents"].apply(_fmt_set)
    rules["rule"] = rules["antecedents_str"] + " => " + rules["consequents_str"]
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

    cols_out = {
        "rule": L("Aturan (Jika => Maka)", "Rule (If => Then)"),
        "antecedents_str": "Antecedent",
        "consequents_str": "Consequent",
        "support": "Support",
        "confidence": "Confidence",
        "lift": "Lift",
        "leverage": "Leverage",
        "conviction": "Conviction",
    }
    have = [c for c in cols_out if c in rules.columns]
    rules_view = rules[have].rename(columns=cols_out).round(4)
    if "zhangs_metric" in rules.columns:
        rules_view["Zhang's Metric"] = rules["zhangs_metric"].round(4)
    rules_view[L("Jumlah Transaksi", "Transaction Count")] = (rules["support"] * len(basket)).round(0).astype(int)
    rules_view[L("Kekuatan", "Strength")] = np.where(
        rules["lift"] >= 3, L("Sangat kuat", "Very strong"),
        np.where(rules["lift"] >= 2, L("Kuat", "Strong"),
                 np.where(rules["lift"] > 1, L("Positif lemah", "Weak positive"),
                          np.where(rules["lift"] == 1, L("Independen", "Independent"),
                                   L("Negatif", "Negative")))))
    tables[L("Aturan Asosiasi", "Association Rules")] = rules_view

    U.metric_row([
        (L("Jumlah aturan", "Rules found"), f"{len(rules):,}"),
        (L("Lift maksimum", "Max lift"), f"{rules['lift'].max():.4f}"),
        (L("Confidence maksimum", "Max confidence"), f"{rules['confidence'].max():.4f}"),
        (L("Support maksimum", "Max support"), f"{rules['support'].max():.4f}"),
    ])
    U.note(L(
        "<b>Support</b>: seberapa sering aturan muncul dalam seluruh transaksi. "
        "<b>Confidence</b>: peluang consequent dibeli bila antecedent dibeli. "
        "<b>Lift &gt; 1</b>: item saling menguatkan (positif); lift = 1 berarti independen; "
        "lift &lt; 1 berarti saling menggantikan.",
        "<b>Support</b>: how often the rule appears across all transactions. "
        "<b>Confidence</b>: probability of the consequent given the antecedent. "
        "<b>Lift &gt; 1</b>: items reinforce each other; lift = 1 means independent; "
        "lift &lt; 1 means substitution.",
    ))
    U.show_table(rules_view.head(100), height=420)

    stat_rules = pd.DataFrame([
        (L("Jumlah aturan", "Number of rules"), len(rules)),
        (L("Rata-rata support", "Mean support"), round(float(rules["support"].mean()), 4)),
        (L("Rata-rata confidence", "Mean confidence"), round(float(rules["confidence"].mean()), 4)),
        (L("Rata-rata lift", "Mean lift"), round(float(rules["lift"].mean()), 4)),
        (L("Median lift", "Median lift"), round(float(rules["lift"].median()), 4)),
        (L("Aturan dengan lift > 1", "Rules with lift > 1"), int((rules["lift"] > 1).sum())),
        (L("Aturan dengan lift > 2", "Rules with lift > 2"), int((rules["lift"] > 2).sum())),
        (L("Aturan dengan confidence > 0,7", "Rules with confidence > 0.7"), int((rules["confidence"] > 0.7).sum())),
    ], columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
    tables[L("Statistik Aturan", "Rule Statistics")] = stat_rules
    U.show_table(stat_rules)

    # -------------------------------------------------------- visualisasi
    U.section(L("4. Visualisasi Aturan", "4. Rule Visualization"))
    figs[L("Support vs confidence", "Support vs confidence")] = V.fig_rules_scatter(rules)
    figs[L("Aturan teratas (lift)", "Top rules (lift)")] = V.fig_rules_bar(rules, "lift")
    figs[L("Aturan teratas (confidence)", "Top rules (confidence)")] = V.fig_rules_bar(rules, "confidence")
    figs[L("Matriks aturan", "Rule matrix")] = V.fig_rules_matrix(rules)
    figs[L("Jaringan aturan", "Rule network")] = V.fig_rules_network(rules)
    for i, (nm, fg) in enumerate(list(figs.items())[-5:]):
        if fg is not None:
            st.markdown(f"**{nm}**")
            U.show_fig(nm, fg, f"ar_viz_{i}")

    # ------------------------------------------- visualisasi khusus algoritma
    U.section(L(f"5. Visualisasi Khusus Algoritma — {cfg['algo']}",
                f"5. Algorithm-Specific Visualization — {cfg['algo']}"),
              L("Grafik berikut memperlihatkan cara algoritma menyusun itemset bertingkat dan "
                "memangkas kandidat yang supportnya di bawah ambang.",
                "The charts below show how the algorithm builds itemsets level by level and prunes "
                "candidates whose support falls below the threshold."))
    U.note(L(
        "Baik Apriori maupun FP-Growth bertumpu pada sifat <b>anti-monoton</b>: sebuah itemset "
        "hanya bisa sering jika seluruh himpunan bagiannya juga sering. Karena itu pencarian "
        "berjalan bertingkat — dari 1-itemset, lalu 2-itemset yang tersusun dari 1-itemset yang "
        "lolos, dan seterusnya. Kisi di bawah menggambarkan hubungan induk-anak tersebut.",
        "Both Apriori and FP-Growth rest on the <b>anti-monotone</b> property: an itemset can only "
        "be frequent if all of its subsets are frequent too. The search therefore proceeds level by "
        "level — from 1-itemsets, then 2-itemsets built from the survivors, and so on. The lattice "
        "below draws those parent-child links."))
    lat = AV.fig_itemset_lattice(freq)
    lev = AV.fig_itemset_levels(freq, cfg["min_sup"])
    quad = AV.fig_rule_quadrant(rules, cfg["min_conf"], cfg["min_lift"])
    for nm, fg in [(L("Kisi itemset", "Itemset lattice"), lat),
                   (L("Itemset per tingkat", "Itemsets per level"), lev),
                   (L("Kuadran mutu aturan", "Rule quality quadrants"), quad)]:
        if fg is not None:
            figs[nm] = fg
            st.markdown(f"**{nm}**")
            st.pyplot(fg, width="stretch")

    # -------------------------------------------------------- interpretasi
    U.section(L("6. Interpretasi Otomatis", "6. Automatic Interpretation"))
    top = rules.iloc[0]
    interp = [
        L(f"Dari {len(basket):,} transaksi dan {basket.shape[1]:,} item unik, algoritma "
          f"**{cfg['algo']}** menemukan **{len(freq):,} itemset sering** (min. support "
          f"{cfg['min_sup']}) dan **{len(rules):,} aturan asosiasi** (min. confidence "
          f"{cfg['min_conf']}, min. lift {cfg['min_lift']}).",
          f"From {len(basket):,} transactions and {basket.shape[1]:,} unique items, "
          f"**{cfg['algo']}** found **{len(freq):,} frequent itemsets** (min. support "
          f"{cfg['min_sup']}) and **{len(rules):,} association rules** (min. confidence "
          f"{cfg['min_conf']}, min. lift {cfg['min_lift']})."),
        L(f"Aturan terkuat: **Jika membeli {top['antecedents_str']}, maka membeli "
          f"{top['consequents_str']}** dengan support {top['support']:.4f} "
          f"({int(top['support'] * len(basket))} transaksi), confidence {top['confidence']:.2%}, "
          f"dan lift {top['lift']:.3f}.",
          f"Strongest rule: **If {top['antecedents_str']} is purchased, then "
          f"{top['consequents_str']} is purchased** with support {top['support']:.4f} "
          f"({int(top['support'] * len(basket))} transactions), confidence {top['confidence']:.2%}, "
          f"and lift {top['lift']:.3f}."),
        L(f"Lift {top['lift']:.3f} berarti peluang membeli {top['consequents_str']} meningkat "
          f"{top['lift']:.2f} kali dibanding peluang dasarnya — indikasi kuat untuk penempatan "
          "produk berdekatan, bundling, atau rekomendasi silang.",
          f"A lift of {top['lift']:.3f} means the chance of buying {top['consequents_str']} is "
          f"{top['lift']:.2f}x its baseline — a strong case for adjacent product placement, "
          "bundling, or cross-recommendation."),
        L(f"Item paling populer: **{item_counts.index[0]}** muncul di "
          f"{item_counts.iloc[0]} transaksi ({item_support.iloc[0]:.2%}); "
          f"rata-rata satu transaksi berisi {sizes.mean():.2f} item.",
          f"Most popular item: **{item_counts.index[0]}** appears in "
          f"{item_counts.iloc[0]} transactions ({item_support.iloc[0]:.2%}); "
          f"an average transaction holds {sizes.mean():.2f} items."),
    ]
    strong = rules[rules["lift"] >= 2]
    if len(strong):
        interp.append(L(
            f"Terdapat {len(strong)} aturan berlift ≥ 2 yang layak ditindaklanjuti, di antaranya: "
            + "; ".join(f"{r['antecedents_str']} → {r['consequents_str']} (lift {r['lift']:.2f})"
                        for _, r in strong.head(3).iterrows()) + ".",
            f"There are {len(strong)} rules with lift ≥ 2 worth acting on, including: "
            + "; ".join(f"{r['antecedents_str']} → {r['consequents_str']} (lift {r['lift']:.2f})"
                        for _, r in strong.head(3).iterrows()) + "."))
    for line in interp:
        st.markdown("- " + line)
    tables[L("Interpretasi", "Interpretation")] = pd.DataFrame(
        {L("No", "No"): range(1, len(interp) + 1),
         L("Interpretasi", "Interpretation"): [t.replace("**", "") for t in interp]})

    meta = {
        L("Proses", "Process"): "Association Rule Mining",
        L("Algoritma", "Algorithm"): cfg["algo"],
        L("Format data", "Data format"): L(*FORMATS[cfg["fmt"]]),
        L("Jumlah transaksi", "Transactions"): len(basket),
        L("Jumlah item", "Items"): basket.shape[1],
        "Min. support": cfg["min_sup"],
        "Min. confidence": cfg["min_conf"],
        "Min. lift": cfg["min_lift"],
        L("Itemset sering", "Frequent itemsets"): len(freq),
        L("Aturan ditemukan", "Rules found"): len(rules),
    }
    U.export_bar(tables, figs, "asosiasi", meta)
