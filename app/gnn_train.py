"""GraphSAGE fraud detection — trains on Account→Account money-flow graph,
writes fraudProb back to Neo4j as a node property."""
import os, sys, time
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)
from rich.table import Table
from rich import box

load_dotenv()

console = Console()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USER"]
PASSWORD = os.environ["NEO4J_PASSWORD"]

EPOCHS     = 150
HIDDEN_DIM = 64
LR         = 0.005
DROPOUT    = 0.3


# ── Model ─────────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return self.conv3(x, edge_index)


# ── Neo4j helpers ──────────────────────────────────────────────────────────────

def load_graph(session):
    rows = session.run("""
        MATCH (a:Account)
        RETURN id(a)                          AS nid,
               a.id                           AS account_id,
               coalesce(a.pageRank,    0.15)  AS pr,
               coalesce(a.betweenness, 0.0)   AS bc,
               coalesce(toFloat(a.wccComponent), 0.0) AS wcc,
               coalesce(CASE WHEN a.flagVelocity THEN 1.0 ELSE 0.0 END, 0.0) AS fv,
               coalesce(CASE WHEN a.flagMule     THEN 1.0 ELSE 0.0 END, 0.0) AS fm,
               coalesce(CASE WHEN a.flagDrain    THEN 1.0 ELSE 0.0 END, 0.0) AS fd
    """).data()

    node_map, account_ids, features = {}, [], []
    for i, r in enumerate(rows):
        node_map[r["nid"]] = i
        account_ids.append(r["account_id"])
        features.append([r["pr"], r["bc"], r["wcc"], r["fv"], r["fm"], r["fd"]])

    label_rows = session.run("""
        MATCH (a:Account)
        OPTIONAL MATCH (a)-[:SENT]->(t1:Transaction)       WHERE t1.isFraud = true
        OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a) WHERE t2.isFraud = true
        WITH a, count(t1) + count(t2) AS fraud_count
        RETURN id(a) AS nid,
               CASE WHEN fraud_count > 0 OR a.flagVelocity OR a.flagMule OR a.flagDrain
                    THEN 1 ELSE 0 END AS label
    """).data()
    labels = [0] * len(rows)
    for r in label_rows:
        if r["nid"] in node_map:
            labels[node_map[r["nid"]]] = r["label"]

    edge_rows = session.run("""
        MATCH (src:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(dst:Account)
        RETURN id(src) AS s, id(dst) AS d
    """).data()
    src_list, dst_list = [], []
    for r in edge_rows:
        if r["s"] in node_map and r["d"] in node_map:
            src_list.append(node_map[r["s"]])
            dst_list.append(node_map[r["d"]])

    return node_map, account_ids, features, labels, src_list, dst_list


def write_fraud_prob(session, account_ids, probs, batch_size=500):
    pairs = [{"id": aid, "p": float(p)} for aid, p in zip(account_ids, probs)]
    with Progress(
        SpinnerColumn(),
        TextColumn("  [cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Writing to Neo4j", total=len(pairs))
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            session.run(
                "UNWIND $rows AS row "
                "MATCH (a:Account {id: row.id}) "
                "SET a.fraudProb = row.p",
                rows=chunk,
            )
            prog.advance(task, len(chunk))


# ── Training helpers ───────────────────────────────────────────────────────────

def make_splits(n, train=0.6, val=0.2):
    idx = torch.randperm(n)
    t   = int(n * train)
    v   = int(n * (train + val))
    return idx[:t], idx[t:v], idx[v:]


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[mask]
        probs  = torch.sigmoid(logits[:, 1]).numpy()
        preds  = (probs >= 0.5).astype(int)
        y_true = data.y[mask].numpy()
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    return dict(
        precision = precision_score(y_true, preds, zero_division=0),
        recall    = recall_score(y_true, preds, zero_division=0),
        f1        = f1_score(y_true, preds, zero_division=0),
        auc       = auc,
        accuracy  = accuracy_score(y_true, preds),
    )


def rule_metrics(data, mask):
    feat   = data.x[mask]
    preds  = ((feat[:, 3] + feat[:, 4] + feat[:, 5]) > 0).numpy().astype(int)
    y_true = data.y[mask].numpy()
    return dict(
        precision = precision_score(y_true, preds, zero_division=0),
        recall    = recall_score(y_true, preds, zero_division=0),
        f1        = f1_score(y_true, preds, zero_division=0),
    )


def ensemble_metrics(data, mask, model):
    model.eval()
    with torch.no_grad():
        gnn_probs = torch.sigmoid(model(data.x, data.edge_index)[mask][:, 1]).numpy()
    feat       = data.x[mask]
    rule_preds = ((feat[:, 3] + feat[:, 4] + feat[:, 5]) > 0).numpy().astype(int)
    gnn_preds  = (gnn_probs >= 0.5).astype(int)
    preds      = np.clip(rule_preds + gnn_preds, 0, 1)
    y_true     = data.y[mask].numpy()
    return dict(
        precision = precision_score(y_true, preds, zero_division=0),
        recall    = recall_score(y_true, preds, zero_division=0),
        f1        = f1_score(y_true, preds, zero_division=0),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]GNN Fraud Detection — GraphSAGE[/bold cyan]\n"
        "[dim]Account→Account money-flow graph  ·  CPU  ·  3 layers  ·  hidden=64  ·  dropout=0.3[/dim]",
        border_style="cyan",
    ))
    console.print()

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    # ── 1 — Load graph ─────────────────────────────────────────────────────────
    console.print("[bold white][[cyan]1/5[/cyan]] Loading graph from Neo4j...[/bold white]")
    t0 = time.perf_counter()
    with driver.session() as session:
        with Progress(SpinnerColumn(), TextColumn("  [cyan]{task.description}"),
                      console=console, transient=True) as prog:
            prog.add_task("Querying accounts + edges...", total=None)
            node_map, account_ids, features, labels, src_list, dst_list = load_graph(session)

    n_nodes   = len(features)
    n_edges   = len(src_list)
    n_fraud   = sum(labels)
    fraud_pct = n_fraud / n_nodes * 100
    console.print(
        f"  [green]✓[/green] [cyan]{n_nodes:,}[/cyan] account nodes  ·  "
        f"[cyan]{n_edges:,}[/cyan] edges  ·  "
        f"[yellow]{n_fraud:,}[/yellow] fraud accounts ([yellow]{fraud_pct:.2f}%[/yellow])  ·  "
        f"[dim]{time.perf_counter()-t0:.1f}s[/dim]"
    )
    console.print()

    # ── 2 — Build PyG Data ─────────────────────────────────────────────────────
    console.print("[bold white][[cyan]2/5[/cyan]] Building PyG graph...[/bold white]")

    x          = torch.tensor(features, dtype=torch.float)
    for col in [0, 1, 2]:   # normalise pageRank, betweenness, wcc to [0,1]
        mn, mx = x[:, col].min(), x[:, col].max()
        x[:, col] = (x[:, col] - mn) / (mx - mn + 1e-8)

    y          = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    data       = Data(x=x, edge_index=edge_index, y=y)

    train_mask, val_mask, test_mask = make_splits(n_nodes)
    pos_weight = (n_nodes - n_fraud) / max(n_fraud, 1)

    console.print(
        f"  Features [cyan](6)[/cyan]: pageRank · betweenness · wccComponent · "
        f"flagVelocity · flagMule · flagDrain"
    )
    console.print(
        f"  Class imbalance → [yellow]pos_weight = {pos_weight:.1f}[/yellow]  "
        f"(weighted cross-entropy)"
    )
    console.print(
        f"  Split: train [cyan]{len(train_mask):,}[/cyan]  "
        f"· val [cyan]{len(val_mask):,}[/cyan]  "
        f"· test [cyan]{len(test_mask):,}[/cyan]"
    )
    console.print()

    # ── 3 — Train ──────────────────────────────────────────────────────────────
    console.print("[bold white][[cyan]3/5[/cyan]] Training GraphSAGE...[/bold white]")

    model = GraphSAGE(data.x.shape[1], HIDDEN_DIM, 2)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    w     = torch.tensor([1.0, float(pos_weight)])

    best_val_auc, best_state = 0.0, None

    with Progress(
        TextColumn("  [cyan]{task.description}"),
        BarColumn(bar_width=38),
        MofNCompleteColumn(),
        TextColumn("[green]loss:[/green] {task.fields[loss]:.4f}"),
        TextColumn("[yellow]val_auc:[/yellow] {task.fields[auc]:.3f}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("epochs", total=EPOCHS, loss=0.0, auc=0.0)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(data.x, data.edge_index)[train_mask],
                                   data.y[train_mask], weight=w)
            loss.backward()
            opt.step()

            if epoch % 5 == 0 or epoch == EPOCHS:
                vm = evaluate(model, data, val_mask)
                if vm["auc"] > best_val_auc:
                    best_val_auc = vm["auc"]
                    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                prog.update(task, advance=5, loss=loss.item(), auc=vm["auc"])
            else:
                prog.update(task, advance=1)

    model.load_state_dict(best_state)
    console.print(
        f"  [green]✓[/green] Best val AUC: [bold cyan]{best_val_auc:.4f}[/bold cyan]"
    )
    console.print()

    # ── 4 — Evaluate ───────────────────────────────────────────────────────────
    console.print("[bold white][[cyan]4/5[/cyan]] Evaluating on test set...[/bold white]")

    gnn_m  = evaluate(model, data, test_mask)
    rule_m = rule_metrics(data, test_mask)
    ens_m  = ensemble_metrics(data, test_mask, model)

    def _color(v):
        return "green" if v >= 0.75 else "yellow" if v >= 0.5 else "red"

    # GNN metrics
    gnn_t = Table(box=box.ROUNDED, border_style="cyan", show_header=True, title="GraphSAGE metrics")
    gnn_t.add_column("Metric",   style="bold white", width=14)
    gnn_t.add_column("Score",    justify="right",    width=10)
    for name, val in gnn_m.items():
        gnn_t.add_row(name.capitalize(), f"[{_color(val)}]{val:.4f}[/{_color(val)}]")
    console.print(gnn_t)
    console.print()

    # Comparison
    comp = Table(
        title="[bold]Method Comparison — test set[/bold]",
        box=box.ROUNDED, border_style="dim white",
    )
    comp.add_column("Method",    style="bold white", width=28)
    comp.add_column("Precision", justify="right",    width=12)
    comp.add_column("Recall",    justify="right",    width=10)
    comp.add_column("F1",        justify="right",    width=10)

    def _fmt(v):
        c = _color(v)
        return f"[{c}]{v:.3f}[/{c}]"

    comp.add_row("Rules only (vel · mule · drain)",
                 _fmt(rule_m["precision"]), _fmt(rule_m["recall"]), _fmt(rule_m["f1"]))
    comp.add_row("GraphSAGE (GNN only)",
                 _fmt(gnn_m["precision"]),  _fmt(gnn_m["recall"]),  _fmt(gnn_m["f1"]))
    comp.add_row("[bold cyan]Rules + GNN ensemble[/bold cyan]",
                 _fmt(ens_m["precision"]),  _fmt(ens_m["recall"]),  _fmt(ens_m["f1"]))
    console.print(comp)
    console.print()

    # ── 5 — Write back ─────────────────────────────────────────────────────────
    console.print(
        "[bold white][[cyan]5/5[/cyan]] Writing [cyan]fraudProb[/cyan] to Neo4j...[/bold white]"
    )
    model.eval()
    with torch.no_grad():
        all_probs = torch.sigmoid(model(data.x, data.edge_index)[:, 1]).numpy().tolist()

    with driver.session() as session:
        write_fraud_prob(session, account_ids, all_probs)

    console.print(
        f"  [green]✓[/green] {n_nodes:,} accounts updated with [cyan]fraudProb[/cyan]"
    )
    console.print()

    # Top 10
    sorted_idx = sorted(range(n_nodes), key=lambda i: all_probs[i], reverse=True)
    top10 = [(account_ids[i], all_probs[i], labels[i], features[i][3:6])
             for i in sorted_idx[:10]]

    top_t = Table(
        title="[bold red]Top 10 Highest-Risk Accounts[/bold red]",
        box=box.ROUNDED, border_style="red",
    )
    top_t.add_column("Account",      style="bold white", width=16)
    top_t.add_column("fraudProb",    justify="right",    width=12)
    top_t.add_column("Ground Truth", justify="center",   width=14)
    top_t.add_column("Rule Flags",   width=28)

    for aid, prob, label, flags in top10:
        fv, fm, fd = int(flags[0]), int(flags[1]), int(flags[2])
        flag_str   = " · ".join(f for f, v in
                                [("velocity", fv), ("mule", fm), ("drain", fd)] if v)
        gt_str     = "[red]FRAUD[/red]" if label else "[dim]clean[/dim]"
        top_t.add_row(
            aid,
            f"[bold red]{prob:.4f}[/bold red]",
            gt_str,
            flag_str or "[dim]—[/dim]",
        )
    console.print(top_t)
    console.print()

    console.print(Panel.fit(
        "[bold green]✓ GNN training complete[/bold green]\n\n"
        "Query high-risk accounts in Neo4j Browser:\n"
        "[cyan]MATCH (a:Account) WHERE a.fraudProb > 0.8\n"
        "RETURN a.id, round(a.fraudProb,4) AS risk\n"
        "ORDER BY risk DESC LIMIT 20[/cyan]",
        border_style="green",
    ))
    console.print()
    driver.close()


if __name__ == "__main__":
    main()
