"""
Variant mapping workflow for DMI Predictor.

Maps curated protein variants from the EBI Proteins API onto predicted
domain-motif interface windows (SLiM match positions and domain match positions).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

API_URL = "https://www.ebi.ac.uk/proteins/api/variation/{}"
HEADERS = {"Accept": "application/json"}


def parse_range(value: str) -> Optional[Tuple[int, int]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    numbers = re.findall(r"\d+", s)
    if not numbers:
        return None
    if len(numbers) == 1:
        pos = int(numbers[0])
        return pos, pos
    start, end = int(numbers[0]), int(numbers[1])
    if start > end:
        start, end = end, start
    return start, end


def extract_variants(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "features" in data and isinstance(data["features"], list):
            return data["features"]
        if "variants" in data and isinstance(data["variants"], list):
            return data["variants"]
    return []


def pick_first(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value)


def extract_clinical_significance(variant: Dict[str, Any]) -> Optional[str]:
    for key in ("clinicalSignificance", "clinicalSignificances"):
        if key in variant:
            val = variant.get(key)
            if isinstance(val, dict):
                return val.get("type")
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, dict):
                    return first.get("type")
                return str(first)
            if val is not None:
                return str(val)
    return None


def extract_consequence(variant: Dict[str, Any]) -> Optional[str]:
    for key in ("consequenceType", "consequenceTypes"):
        if key in variant:
            return pick_first(variant.get(key))
    return None


def extract_xrefs(variant: Dict[str, Any]) -> str:
    xrefs = variant.get("xrefs") or variant.get("crossReferences") or []
    if not isinstance(xrefs, list):
        return ""
    ids = []
    for xref in xrefs:
        if not isinstance(xref, dict):
            continue
        name = str(xref.get("name") or xref.get("dbName") or "").lower()
        xid = xref.get("id") or xref.get("primaryId") or xref.get("value")
        if xid is None:
            continue
        if "dbsnp" in name or "clinvar" in name or "rs" in str(xid).lower():
            ids.append(str(xid))
    return ";".join(sorted(set(ids)))


def fetch_variants(session: requests.Session, accession: str, delay: float) -> List[Dict[str, Any]]:
    url = API_URL.format(accession)
    resp = session.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    time.sleep(delay)
    data = resp.json()
    return extract_variants(data)


def build_windows_by_protein(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    windows_by_protein: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        intx_id = row.get("intx_ID")
        slim_prot = row.get("SLiMProtein")
        slim_match = row.get("SLiMMatch")
        domain_prot = row.get("DomainProtein")
        domain_match = row.get("DomainMatch1")
        elm = row.get("Elm")
        domain_name1 = row.get("DomainName1")

        dmi_id = f"{intx_id}|{elm}|{slim_match}|{domain_name1}|{domain_match}"
        dmi_row_dict = row.to_dict()
        dmi_row_dict["DMI_ID"] = dmi_id

        slim_range = parse_range(slim_match)
        if slim_prot and slim_range:
            accession = str(slim_prot).strip()
            windows_by_protein.setdefault(accession, []).append(
                {
                    **dmi_row_dict,
                    "protein": accession,
                    "window_type": "SLiM",
                    "window_raw": str(slim_match),
                    "start": slim_range[0],
                    "end": slim_range[1],
                }
            )

        if domain_prot and domain_match and not (isinstance(domain_match, float) and pd.isna(domain_match)):
            accession = str(domain_prot).strip()
            for interval in str(domain_match).split("|"):
                domain_range = parse_range(interval)
                if domain_range:
                    windows_by_protein.setdefault(accession, []).append(
                        {
                            **dmi_row_dict,
                            "protein": accession,
                            "window_type": "Domain",
                            "window_raw": interval.strip(),
                            "start": domain_range[0],
                            "end": domain_range[1],
                        }
                    )

    return windows_by_protein


def map_variants_to_windows(
    windows: Iterable[Dict[str, Any]],
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    if not variants:
        return mapped
    for window in windows:
        start = int(window["start"])
        end = int(window["end"])
        for var in variants:
            begin = var.get("begin") or var.get("start")
            if begin is None:
                continue
            try:
                pos = int(begin)
            except (TypeError, ValueError):
                continue
            if start <= pos <= end:
                merged = dict(window)
                merged.update({
                    "window_start": start,
                    "window_end": end,
                    "position": pos,
                    "wild_type": var.get("wildType"),
                    "mutant": var.get("alternativeSequence") or var.get("mutatedType"),
                    "variant_type": var.get("type"),
                    "clinical_significance": extract_clinical_significance(var),
                    "consequence_type": extract_consequence(var),
                    "xrefs": extract_xrefs(var),
                })
                mapped.append(merged)
    return mapped


def run_variant_mapping(
    input_file: str,
    output_file: str,
    dmi_score_cutoff: Optional[float],
    delay: float,
    verbose: bool,
) -> None:
    df = pd.read_csv(input_file, sep="\t")

    if dmi_score_cutoff is not None:
        if "DMIMatchScore" not in df.columns:
            raise ValueError("Input has no 'DMIMatchScore' column to filter on.")
        before = len(df)
        df = df[pd.to_numeric(df["DMIMatchScore"], errors="coerce") > dmi_score_cutoff]
        if verbose:
            print(f"DMIMatchScore > {dmi_score_cutoff}: kept {len(df)}/{before} rows.")

    windows_by_protein = build_windows_by_protein(df)
    if not windows_by_protein:
        if verbose:
            print("No valid windows found in input.")
        return

    unique_proteins = sorted(windows_by_protein.keys())
    if verbose:
        print(f"Querying EBI Proteins API for {len(unique_proteins)} unique proteins...")

    session = requests.Session()
    not_found = 0
    total_mapped = 0
    proteins_with_variants: set = set()
    wrote_header = False

    for accession in tqdm(unique_proteins, desc="Fetching variants", disable=not verbose):
        try:
            variants = fetch_variants(session, accession, delay)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                variants = []
                not_found += 1
            else:
                raise

        mapped = map_variants_to_windows(windows_by_protein[accession], variants)
        if not mapped:
            continue

        out_df = pd.DataFrame(mapped)
        out_df.to_csv(output_file, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        total_mapped += len(mapped)
        proteins_with_variants.add(accession)

    if verbose:
        if total_mapped == 0:
            print("No variants mapped to any windows.")
        else:
            print(
                f"Mapped {total_mapped} variants across {len(proteins_with_variants)} proteins "
                f"(not found: {not_found})."
            )
            print(f"Saved output to {output_file}.")
