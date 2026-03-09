"""
Domain position correction via InterPro REST API.

For every unique (DomainProtein, DomainID) pair in a DMI prediction TSV,
queries InterPro to retrieve current domain boundary positions on that protein,
then overwrites DomainMatch1 / DomainMatch2 with the InterPro coordinates.

Original positions are preserved in DomainMatch1_DMI / DomainMatch2_DMI.
Results are cached to a JSON file to avoid redundant API calls on re-runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_file: Path) -> None:
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


# ── InterPro query ─────────────────────────────────────────────────────────────

def _domain_id_to_db(domain_id: str) -> str:
    if domain_id.startswith("PF"):
        return "pfam"
    elif domain_id.startswith("SM"):
        return "smart"
    else:
        raise ValueError(f"Unknown domain ID prefix: {domain_id}")


def _query_interpro(
    uniprot_id: str,
    domain_id: str,
    cache: dict,
    api_delay: float,
    retries: int = 3,
) -> Optional[str]:
    """
    Fetch domain positions for (uniprot_id, domain_id) from InterPro.

    Returns a position string ('start-end' or 'start1-end1|start2-end2|...')
    or None if the domain is not found on this protein.
    Results are stored in cache using key 'UniProtID::DomainID'.
    """
    key = f"{uniprot_id}::{domain_id}"
    if key in cache:
        return cache[key]

    try:
        db = _domain_id_to_db(domain_id)
    except ValueError:
        cache[key] = None
        return None

    url = (
        f"{INTERPRO_BASE}/entry/{db}/{domain_id}"
        f"/protein/uniprot/{uniprot_id}/?format=json"
    )

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20)
        except requests.exceptions.RequestException as exc:
            print(f"    Network error ({uniprot_id}, {domain_id}): {exc}", flush=True)
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 404:
            cache[key] = None
            return None

        if resp.status_code in (408, 429):
            wait = 2 ** attempt * 2
            print(f"    HTTP {resp.status_code} — waiting {wait}s ...", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for ({uniprot_id}, {domain_id})", flush=True)
            cache[key] = None
            return None

        data = resp.json()
        proteins = data.get("proteins", [])
        if not proteins:
            cache[key] = None
            return None

        locations = proteins[0].get("entry_protein_locations", [])
        if not locations:
            cache[key] = None
            return None

        ranges = []
        for loc in locations:
            frags = loc.get("fragments", [])
            if not frags:
                continue
            loc_start = frags[0]["start"]
            loc_end = frags[-1]["end"]
            ranges.append(f"{loc_start}-{loc_end}")

        result = "|".join(ranges) if ranges else None
        cache[key] = result
        time.sleep(api_delay)
        return result

    print(f"    All retries failed for ({uniprot_id}, {domain_id})", flush=True)
    cache[key] = None
    return None


# ── Main entry point ───────────────────────────────────────────────────────────

def run_domain_correction(
    input_file: str,
    output_file: str,
    cache_file: Optional[str] = None,
    api_delay: float = 0.25,
    verbose: bool = False,
) -> None:
    """
    Read a DMI prediction TSV, correct DomainMatch1/DomainMatch2 positions
    using current InterPro data, and write the corrected file.

    Original positions are preserved in DomainMatch1_DMI / DomainMatch2_DMI.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Default cache file alongside the output
    if cache_file is None:
        cache_path = output_path.parent / "interpro_cache.json"
    else:
        cache_path = Path(cache_file)

    if verbose:
        print(f"Loading predictions from {input_path.name} ...")
    df = pd.read_csv(input_path, sep="\t")
    if verbose:
        print(f"  {len(df):,} rows")

    # Preserve originals
    df["DomainMatch1_DMI"] = df["DomainMatch1"]
    df["DomainMatch2_DMI"] = df["DomainMatch2"]

    # Collect unique (DomainProtein, DomainID) pairs for both domain slots
    lookup_cols = [
        ("DomainProtein", "DomainID1", "DomainMatch1"),
        ("DomainProtein", "DomainID2", "DomainMatch2"),
    ]
    unique_pairs: dict[tuple, set] = {}
    for prot_col, id_col, match_col in lookup_cols:
        for _, row in df[[prot_col, id_col]].dropna().drop_duplicates().iterrows():
            key = (row[prot_col], row[id_col])
            unique_pairs.setdefault(key, set()).add(match_col)

    total = len(unique_pairs)
    cache = _load_cache(cache_path)
    already_cached = sum(
        1 for (up, did) in unique_pairs if f"{up}::{did}" in cache
    )

    if verbose:
        print(
            f"\n{total} unique (DomainProtein, DomainID) pairs — "
            f"{already_cached} cached, {total - already_cached} new API calls"
        )

    # Query InterPro
    results: dict[tuple, Optional[str]] = {}
    n_found = n_missing = 0

    for i, (uniprot_id, domain_id) in enumerate(unique_pairs, 1):
        pos = _query_interpro(uniprot_id, domain_id, cache, api_delay)
        results[(uniprot_id, domain_id)] = pos
        if pos:
            n_found += 1
        else:
            n_missing += 1
        if verbose:
            status = "cached" if f"{uniprot_id}::{domain_id}" in cache else ("found" if pos else "not found")
            print(
                f"  [{i:>3}/{total}] {uniprot_id:12s} {domain_id:10s} → "
                f"{pos if pos else '—':35s} ({status})",
                flush=True,
            )
        if i % 20 == 0:
            _save_cache(cache, cache_path)

    _save_cache(cache, cache_path)

    if verbose:
        print(f"\nLookup complete: {n_found} found, {n_missing} not found")

    # Apply corrections
    updated1 = updated2 = 0
    for idx, row in df.iterrows():
        up = row.get("DomainProtein")
        did1 = row.get("DomainID1")
        did2 = row.get("DomainID2")

        if pd.notna(up) and pd.notna(did1):
            new_pos = results.get((up, did1))
            if new_pos is not None and new_pos != row["DomainMatch1"]:
                df.at[idx, "DomainMatch1"] = new_pos
                updated1 += 1

        if pd.notna(up) and pd.notna(did2):
            new_pos2 = results.get((up, did2))
            if new_pos2 is not None and new_pos2 != row["DomainMatch2"]:
                df.at[idx, "DomainMatch2"] = new_pos2
                updated2 += 1

    # Move _DMI backup columns next to their originals
    cols = list(df.columns)
    for orig, backup in [("DomainMatch1", "DomainMatch1_DMI"), ("DomainMatch2", "DomainMatch2_DMI")]:
        if backup in cols:
            cols.remove(backup)
            cols.insert(cols.index(orig) + 1, backup)
    df = df[cols]

    df.to_csv(output_path, sep="\t", index=False)

    print(
        f"Domain correction complete: DomainMatch1 {updated1} rows updated, "
        f"DomainMatch2 {updated2} rows updated. "
        f"Cache: {cache_path}"
    )
