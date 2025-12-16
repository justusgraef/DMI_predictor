"""
DMI Predictor core workflow module.

Implements the end-to-end prediction pipeline:
- Loads reference data
- Loads model and imputer
- Reads protein sequences and PPI pairs
- Extracts features (with hooks for IUPred2a integration)
- Performs domain-motif matching
- Scores candidates with Random Forest
- Outputs results
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
import joblib
import numpy as np

from dmi_predictor.config import DMIPredictorConfig
from dmi_predictor.io.result_writer import ResultWriter
from dmi_predictor.core.dmidb import InterfaceHandling


FEATURE_ORDER = [
    "Probability",
    "IUPredShort",
    "Anchor",
    "DomainOverlap",
    "qfo_RLC",
    "qfo_RLCvar",
    "vertebrates_RLC",
    "vertebrates_RLCvar",
    "mammalia_RLC",
    "mammalia_RLCvar",
    "metazoa_RLC",
    "metazoa_RLCvar",
    "DomainEnrichment_pvalue",
    "DomainEnrichment_zscore",
    "DomainFreqbyProtein",
    "DomainFreqinProteome",
]


class DMIWorkflow:
    def __init__(
        self,
        config: DMIPredictorConfig,
        verbose: bool = False,
        features_dir: Optional[str] = None,
        skip_elm: bool = False,
    ):
        self.config = config
        self.verbose = verbose
        self.features_dir = Path(features_dir) if features_dir else None
        self.skip_elm = skip_elm
        self.model = joblib.load(config.model_file)
        self.imputer = joblib.load(config.imputer_file)
        # Set module-level switch in dmidb to control ELM calls
        try:
            from dmi_predictor.core import dmidb as _dmidb

            _dmidb.SKIP_ELM = skip_elm
        except Exception:
            pass

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _init_interface(self, sequences: Dict[str, str], ppi_pairs: List[Tuple[str, str]]):
        iface = InterfaceHandling(
            prot_path="",
            slim_type_file=str(self.config.elm_classes_file),
            dmi_type_file=str(self.config.elm_dmi_file),
            smart_domain_types_file=str(self.config.smart_freq_file),
            pfam_domain_types_file=str(self.config.pfam_freq_file),
            smart_domain_matches_json_file=str(self.config.interpro_smart_file),
            pfam_domain_matches_json_file=str(self.config.interpro_pfam_file),
            features_path=str(self.features_dir) if self.features_dir else None,
            network_path=str(self.features_dir) if self.features_dir else None,
        )
        iface.load_sequences_from_dict(sequences)
        iface.set_protein_pairs(ppi_pairs)
        iface.read_in_slim_types()
        iface.read_in_DMI_types()
        iface.read_in_domain_types()
        iface.read_in_domain_matches()
        iface.read_in_features_all_proteins()
        iface.read_in_networks(prot_set=set(sequences.keys()))
        return iface

    def _create_slim_matches_restricted(self, iface: InterfaceHandling):
        for protpair in iface.protein_pairs_dict:
            protein_pairs = [(protpair[0], protpair[1]), (protpair[1], protpair[0])]
            for protein_pair in protein_pairs:
                prot_inst = iface.proteins_dict[protein_pair[0]]
                domain_matches_dict = prot_inst.domain_matches_dict
                sel_dmi_type_inst = []
                for _, dmi_type_inst in iface.dmi_types_dict.items():
                    domain_interface = dmi_type_inst.domain_interfaces
                    if len(domain_interface) == 1:
                        domains = set(dmi_type_inst.domain_interfaces[0].domain_dict.keys())
                        if domains.intersection(set(domain_matches_dict.keys())) == domains:
                            sel_dmi_type_inst.append((dmi_type_inst, domains))
                    else:
                        domains = set()
                        for domain_int in domain_interface:
                            for domain_id in domain_int.domain_dict.keys():
                                domains.add(domain_id)
                            sel_domain = domains.intersection(set(domain_matches_dict.keys()))
                            if any(sel_domain):
                                sel_dmi_type_inst.append((dmi_type_inst, sel_domain))
                for dmi_type_inst, domain_type_list in sel_dmi_type_inst:
                    slim_type_inst = iface.slim_types_dict[dmi_type_inst.slim_id]
                    iface.proteins_dict[protein_pair[1]].create_slim_matches(dmi_type_inst, slim_type_inst)
                    if dmi_type_inst.slim_id in iface.proteins_dict[protein_pair[1]].slim_matches_dict:
                        for slim_match_inst in iface.proteins_dict[protein_pair[1]].slim_matches_dict[dmi_type_inst.slim_id]:
                            slim_match_inst.get_slim_match_features(domain_type_list)

    def _score_dmi_matches(self, iface: InterfaceHandling):
        for _, prot_pair_inst in iface.protein_pairs_dict.items():
            for slim_id, dmi_match_inst_list in prot_pair_inst.dmi_matches_dict.items():
                for dmi_match_inst in dmi_match_inst_list:
                    feature_array = []
                    slim_type = iface.slim_types_dict[slim_id]
                    try:
                        probability_value = float(slim_type.probability)
                    except (TypeError, ValueError):
                        probability_value = np.nan
                    feature_array.append(probability_value)
                    slim_match_inst = dmi_match_inst.slim_match
                    features_dict = slim_match_inst.__dict__
                    for feature in [
                        "IUPredShort",
                        "Anchor",
                        "DomainOverlap",
                        "qfo_RLC",
                        "qfo_RLCvar",
                        "vertebrates_RLC",
                        "vertebrates_RLCvar",
                        "mammalia_RLC",
                        "mammalia_RLCvar",
                        "metazoa_RLC",
                        "metazoa_RLCvar",
                        "DomainEnrichment_pvalue",
                        "DomainEnrichment_zscore",
                    ]:
                        value = features_dict.get(feature)
                        if value is None:
                            feature_array.append(np.nan)
                        else:
                            feature_array.append(value)

                    domainfreqsbyprotein = []
                    domainfreqsinproteome = []
                    for domain_match_list in dmi_match_inst.domain_interface_match.domain_matches:
                        domain_id = domain_match_list[0].domain_id
                        try:
                            domainfreqsbyprotein.append(float(iface.domain_types_dict[domain_id].DomainFreqbyProtein))
                        except (TypeError, ValueError):
                            domainfreqsbyprotein.append(np.nan)
                        try:
                            domainfreqsinproteome.append(float(iface.domain_types_dict[domain_id].DomainFreqinProteome))
                        except (TypeError, ValueError):
                            domainfreqsinproteome.append(np.nan)
                    if len(domainfreqsbyprotein) > 1:
                        feature_array.append(float(np.mean(domainfreqsbyprotein)))
                        feature_array.append(float(np.mean(domainfreqsinproteome)))
                    else:
                        feature_array.extend(domainfreqsbyprotein)
                        feature_array.extend(domainfreqsinproteome)

                    feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
                    transformed_feature_array = self.imputer.transform(feature_array_np)
                    missing_indicator = np.isnan(feature_array_np)
                    dmi_match_inst.score = self.model.predict_proba(transformed_feature_array)[:, 1][0]
                    missing_mask = missing_indicator.ravel()
                    missing = [FEATURE_ORDER[i] for i, miss in enumerate(missing_mask) if miss]
                    dmi_match_inst.missing_feature = ",".join(missing) if missing else ""

    def _build_results(self, iface: InterfaceHandling, score_threshold: float) -> List[Dict[str, Any]]:
        rows = []
        for prot_pair, prot_pair_inst in iface.protein_pairs_dict.items():
            intx_ID = "_".join([id for id in prot_pair])
            for slim_id, dmi_match_inst_list in prot_pair_inst.dmi_matches_dict.items():
                for dmi_match_inst in dmi_match_inst_list:
                    if dmi_match_inst.score is None or dmi_match_inst.score < score_threshold:
                        continue
                    slim_match = dmi_match_inst.slim_match
                    slim_type = slim_match.slim_type_inst
                    row = {
                        "intx_ID": intx_ID,
                        "Accession": slim_type.slim_id,
                        "Elm": slim_type.name,
                        "Regex": slim_type.regex,
                        "Pattern": slim_match.pattern,
                        "Probability": slim_type.probability,
                        "SLiMProtein": dmi_match_inst.slim_protein,
                        "SLiMMatch": f"{slim_match.start}-{slim_match.end}",
                        "IUPredShort": slim_match.IUPredShort,
                        "Anchor": slim_match.Anchor,
                        "DomainOverlap": slim_match.DomainOverlap,
                        "qfo_RLC": slim_match.qfo_RLC,
                        "qfo_RLCvar": slim_match.qfo_RLCvar,
                        "vertebrates_RLC": slim_match.vertebrates_RLC,
                        "vertebrates_RLCvar": slim_match.vertebrates_RLCvar,
                        "mammalia_RLC": slim_match.mammalia_RLC,
                        "mammalia_RLCvar": slim_match.mammalia_RLCvar,
                        "metazoa_RLC": slim_match.metazoa_RLC,
                        "metazoa_RLCvar": slim_match.metazoa_RLCvar,
                        "DomainEnrichment_pvalue": slim_match.DomainEnrichment_pvalue,
                        "DomainEnrichment_zscore": slim_match.DomainEnrichment_zscore,
                        "Partner_with_domain": ",".join([p.protein_id for p in slim_match.partners_with_domain_in_real_network]),
                        "DomainProtein": dmi_match_inst.domain_protein,
                        "DMIMatchScore": dmi_match_inst.score,
                        "Notes": "",
                    }

                    domain_fields = [
                        "DomainID1",
                        "DomainName1",
                        "DomainMatch1",
                        "DomainMatchFound1",
                        "DomainMatchRequired1",
                        "DomainMatchEvalue1",
                        "DomainFreqbyProtein1",
                        "DomainFreqinProteome1",
                        "DomainID2",
                        "DomainName2",
                        "DomainMatch2",
                        "DomainMatchFound2",
                        "DomainMatchRequired2",
                        "DomainMatchEvalue2",
                        "DomainFreqbyProtein2",
                        "DomainFreqinProteome2",
                    ]
                    for f in domain_fields:
                        row[f] = ""

                    for idx, domain_match_list in enumerate(dmi_match_inst.domain_interface_match.domain_matches, start=1):
                        domain_id = domain_match_list[0].domain_id
                        domain_name = iface.domain_types_dict[domain_id].name
                        start_list = [dm.start for dm in domain_match_list]
                        end_list = [dm.end for dm in domain_match_list]
                        match = "|".join([f"{s}-{e}" for s, e in zip(start_list, end_list)])
                        evalues = "|".join([str(dm.evalue) for dm in domain_match_list])
                        domain_count_found = len(domain_match_list)
                        domain_count_required = None
                        for intf in iface.dmi_types_dict[slim_id].domain_interfaces:
                            if domain_id in intf.domain_dict:
                                domain_count_required = intf.domain_dict[domain_id]
                        row[f"DomainID{idx}"] = domain_id
                        row[f"DomainName{idx}"] = domain_name
                        row[f"DomainMatch{idx}"] = match
                        row[f"DomainMatchFound{idx}"] = domain_count_found
                        row[f"DomainMatchRequired{idx}"] = domain_count_required
                        row[f"DomainMatchEvalue{idx}"] = evalues
                        row[f"DomainFreqbyProtein{idx}"] = iface.domain_types_dict[domain_id].DomainFreqbyProtein
                        row[f"DomainFreqinProteome{idx}"] = iface.domain_types_dict[domain_id].DomainFreqinProteome

                    if len(iface.dmi_types_dict[slim_id].domain_interfaces) > 1:
                        domain_required = []
                        for intf in iface.dmi_types_dict[slim_id].domain_interfaces:
                            domain_required.extend([(d, count) for d, count in intf.domain_dict.items()])
                        row["Notes"] = f"Requires additional partner domain(s): {domain_required}"
                    if dmi_match_inst.missing_feature:
                        note = f"Missing features imputed: {dmi_match_inst.missing_feature}"
                        row["Notes"] = f"{row['Notes']} | {note}" if row["Notes"] else note

                    rows.append(row)
        return rows

    def _run_chunk(
        self,
        ppi_pairs: List[Tuple[str, str]],
        sequences: Dict[str, str],
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        iface = self._init_interface(sequences, ppi_pairs)
        self._log("Running SLiM matching...")
        self._create_slim_matches_restricted(iface)
        self._log("Finding DMI matches...")
        iface.find_DMI_matches()
        self._log("Scoring matches...")
        self._score_dmi_matches(iface)
        return self._build_results(iface, score_threshold)

    def run(
        self,
        ppi_pairs: List[Tuple[str, str]],
        sequences: Dict[str, str],
        output_file: str,
        output_format: str = "tsv",
        score_threshold: float = 0.0,
        num_workers: int = 1,
    ) -> None:
        if num_workers is None or num_workers < 1:
            num_workers = 1

        # Serial path
        if num_workers == 1 or len(ppi_pairs) == 1:
            results = self._run_chunk(ppi_pairs, sequences, score_threshold)
        else:
            max_workers = min(num_workers, len(ppi_pairs))
            chunk_size = (len(ppi_pairs) + max_workers - 1) // max_workers
            chunks = [ppi_pairs[i : i + chunk_size] for i in range(0, len(ppi_pairs), chunk_size)]

            # Parallel execution per PPI chunk
            futures = []
            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for chunk in chunks:
                    futures.append(
                        executor.submit(
                            _predict_chunk_task,
                            chunk,
                            sequences,
                            str(self.config.data_dir),
                            self.features_dir,
                            self.verbose,
                            score_threshold,
                            self.skip_elm,
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    results.extend(fut.result())

        if not results:
            self._log("No matches passed the score threshold.")
            return

        if output_format == "tsv":
            ResultWriter.write_tsv(results, output_file)
        elif output_format == "csv":
            ResultWriter.write_csv(results, output_file)
        elif output_format == "json":
            ResultWriter.write_json(results, output_file)
        else:
            raise ValueError(f"Unknown output format: {output_format}")


def _predict_chunk_task(ppi_pairs_chunk, sequences, data_dir, features_dir, verbose, score_threshold, skip_elm):
    cfg = DMIPredictorConfig(data_dir=data_dir)
    workflow = DMIWorkflow(cfg, verbose=verbose, features_dir=features_dir, skip_elm=skip_elm)
    return workflow._run_chunk(ppi_pairs_chunk, sequences, score_threshold)


def run_dmi_prediction(ppi_pairs, sequences, config, output_file, output_format, score_threshold, verbose, features_dir=None, skip_elm: bool = False, num_workers: int = 1):
    workflow = DMIWorkflow(config, verbose=verbose, features_dir=features_dir, skip_elm=skip_elm)
    workflow.run(ppi_pairs, sequences, output_file, output_format, score_threshold, num_workers=num_workers)
