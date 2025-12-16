"""
Port of DMIDB classes for DMI prediction.
"""
import re
import json
import glob
import numpy as np
import scipy.special as sc

from . import protein_interaction_interfaces
from .protein_interaction_interfaces import (
    Protein as BaseProtein,
    ProteinPair as BaseProteinPair,
    DomainType as BaseDomainType,
    DomainMatch,
    DomainInterface,
    DomainInterfaceMatch as BaseDomainInterfaceMatch,
    InterfaceHandling as BaseInterfaceHandling,
)
import requests

# Module-level switch to skip ELM network calls (set by workflow/CLI)
SKIP_ELM = False

dummy_value = 88888

class Protein(BaseProtein):
    def __init__(self, protein_id):
        super().__init__(protein_id)
        self.slim_matches_dict = {}
        self.IUPredShort_scores = []
        self.Anchor_scores = []
        self.DomainOverlap_scores = []
        self.qfo_RLC_scores = {}
        self.vertebrates_RLC_scores = {}
        self.mammalia_RLC_scores = {}
        self.metazoa_RLC_scores = {}
        self.networks = {}
        self.network_degree = None

    def create_slim_matches(self, dmi_type_inst, slim_type_inst):
        slim_start = [match.start() for match in re.finditer('(?=(' + slim_type_inst.regex + '))', self.sequence)]
        match_pattern = [match.group(1) for match in re.finditer('(?=(' + slim_type_inst.regex + '))', self.sequence)]
        match_results = list(zip(slim_start, match_pattern))
        if len(match_results) > 0:
            self.slim_matches_dict[slim_type_inst.slim_id] = []
            for match in match_results:
                if match[0] == 0:
                    pattern = self.sequence[match[0] : match[0] + len(match[1]) + 1]
                    modified_pattern = '-' + pattern[:-1] + str.lower(pattern[-1])
                elif match[0] + len(match[1]) == len(self.sequence):
                    pattern = self.sequence[match[0] - 1 : match[0] + len(match[1])]
                    modified_pattern = str.lower(pattern[0]) + pattern[1:]
                else:
                    pattern = self.sequence[match[0] - 1 : match[0] + len(match[1]) + 1]
                    modified_pattern = str.lower(pattern[0]) + pattern[1:-1] + str.lower(pattern[-1])
                slim_match_inst = SLiMMatch(
                    dmi_type_inst,
                    slim_type_inst,
                    self,
                    match[0] + 1,
                    match[0] + len(match[1]),
                    modified_pattern,
                )
                self.slim_matches_dict[slim_type_inst.slim_id].append(slim_match_inst)

    def read_in_features(self, features_path):
        try:
            with open(features_path + '/IUPred_short/' + self.protein_id + '_iupredshort.txt', 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            for line in lines[1:]:
                self.IUPredShort_scores.append(float(line.split('\t')[2]))
        except FileNotFoundError:
            pass
        try:
            with open(features_path + '/Anchor/' + self.protein_id + '_anchor.txt', 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            for line in lines[1:]:
                self.Anchor_scores.append(float(line.split('\t')[2]))
        except FileNotFoundError:
            pass
        try:
            with open(features_path + '/Domain_overlap/' + self.protein_id + '_domain_overlap.txt', 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            for line in lines[1:]:
                self.DomainOverlap_scores.append(float(line.split('\t')[2]))
        except FileNotFoundError:
            pass
        try:
            with open(features_path + '/conservation_scores/' + self.protein_id + '_con.json', 'r') as f:
                data = json.load(f)
            for result in data['Conservation']:
                if 'qfo' in result:
                    self.qfo_RLC_scores = result['qfo']
                elif 'vertebrates' in result:
                    self.vertebrates_RLC_scores = result['vertebrates']
                elif 'mammalia' in result:
                    self.mammalia_RLC_scores = result['mammalia']
                elif 'metazoa' in result:
                    self.metazoa_RLC_scores = result['metazoa']
        except FileNotFoundError:
            pass

    def calculate_features_scores(self):
        for slim_id, slim_match in self.slim_matches_dict.items():
            for slim_match_inst in slim_match:
                slim_match_inst.get_slim_match_features()

class ProteinPair(BaseProteinPair):
    def __init__(self, proteinA, proteinB):
        super().__init__(proteinA, proteinB)
        self.dmi_matches_dict = {}

class SLiMType:
    def __init__(self, slim_id):
        self.slim_id = slim_id
        self.name = ''
        self.regex = ''
        self.probability = ''

class SLiMMatch:
    def __init__(self, dmi_type_inst, slim_type_inst, prot_inst, start, end, pattern):
        self.dmi_type_inst = dmi_type_inst
        self.slim_type_inst = slim_type_inst
        self.prot_inst = prot_inst
        self.start = start
        self.end = end
        self.pattern = pattern
        self.IUPredShort = None
        self.Anchor = None
        self.DomainOverlap = None
        self.qfo_RLC = None
        self.qfo_RLCvar = None
        self.vertebrates_RLC = None
        self.vertebrates_RLCvar = None
        self.mammalia_RLC = None
        self.mammalia_RLCvar = None
        self.metazoa_RLC = None
        self.metazoa_RLCvar = None
        self.DomainEnrichment_pvalue = None
        self.DomainEnrichment_zscore = None
        self.vertex_with_domain_in_real_network = None
        self.partners_with_domain_in_real_network = set()

    def get_slim_match_features(self, domain_type_list=None):
        defined_positions_url = 'http://slim.icr.ac.uk/restapi/functions/defined_positions?'
        start = int(self.start)
        end = int(self.end)
        pattern = self.pattern
        regex = self.slim_type_inst.regex
        if self.prot_inst.IUPredShort_scores:
            self.IUPredShort = float(sum(self.prot_inst.IUPredShort_scores[start-1:end])/(end - start + 1))
        else:
            self.IUPredShort = 0.0
        if self.prot_inst.Anchor_scores:
            self.Anchor = float(sum(self.prot_inst.Anchor_scores[start-1:end])/(end - start + 1))
        else:
            self.Anchor = 0.0
        if self.prot_inst.DomainOverlap_scores:
            self.DomainOverlap = float(sum(self.prot_inst.DomainOverlap_scores[start-1:end])/(end - start + 1))
        else:
            self.DomainOverlap = 0.0
        # Direct ELM API call (no caching)
        if not SKIP_ELM:
            try:
                response = requests.get(
                    "http://slim.icr.ac.uk/restapi/functions/defined_positions",
                    params={"motif": regex, "sequence": pattern},
                    timeout=10
                )
                if response.status_code == 200:
                    response = response.json()
                else:
                    response = None
            except Exception:
                response = None
        else:
            response = None
        
        if not response:
            self._set_cons_problem()
        else:
            defined_positions = [start + (ind - 1) for ind in response.get("indexes", [])]
            for i, cons_type in enumerate([
                self.prot_inst.qfo_RLC_scores,
                self.prot_inst.vertebrates_RLC_scores,
                self.prot_inst.mammalia_RLC_scores,
                self.prot_inst.metazoa_RLC_scores,
            ]):
                if any(cons_type):
                    defined_positions_cons_scores = []
                    for pos, score in cons_type.items():
                        if int(pos) in defined_positions:
                            defined_positions_cons_scores.append(score)
                    if any(defined_positions_cons_scores):
                        pmotif = np.product(defined_positions_cons_scores)
                        lnpmotif = -np.log(pmotif)
                        sigmotif = sc.gammaincc(len(defined_positions_cons_scores), lnpmotif)
                        meanRLCprob = np.mean(defined_positions_cons_scores)
                        varRLCprob = sum([abs(x - meanRLCprob) for x in defined_positions_cons_scores]) / len(defined_positions_cons_scores)
                        if i == 0:
                            self.qfo_RLC = sigmotif
                            self.qfo_RLCvar = varRLCprob
                        elif i == 1:
                            self.vertebrates_RLC = sigmotif
                            self.vertebrates_RLCvar = varRLCprob
                        elif i == 2:
                            self.mammalia_RLC = sigmotif
                            self.mammalia_RLCvar = varRLCprob
                        elif i == 3:
                            self.metazoa_RLC = sigmotif
                            self.metazoa_RLCvar = varRLCprob
        if any(self.prot_inst.networks):
            num_rand_networks = len(self.prot_inst.networks) - 1
            vertices_with_overlapping_domains = {}
            if domain_type_list is None:
                domain_match_list = self.dmi_type_inst.domain_interfaces
                if len(domain_match_list) == 1:
                    domains = set(self.dmi_type_inst.domain_interfaces[0].domain_dict.keys())
                    for network_id, network in self.prot_inst.networks.items():
                        count = 0
                        for partner in network:
                            partner_domains = set(partner.domain_matches_dict.keys())
                            if domains.intersection(partner_domains) == domains:
                                count += 1
                                if network_id == 0:
                                    self.partners_with_domain_in_real_network.add(partner)
                        vertices_with_overlapping_domains[int(network_id)] = count
                else:
                    domains = set()
                    for domain_intf_obj in domain_match_list:
                        for domain_id in domain_intf_obj.domain_dict.keys():
                            domains.add(domain_id)
                    for network_id, network in self.prot_inst.networks.items():
                        count = 0
                        for partner in network:
                            partner_domains = set(partner.domain_matches_dict.keys())
                            if any(domains.intersection(partner_domains)):
                                count += 1
                                if network_id == 0:
                                    self.partners_with_domain_in_real_network.add(partner)
                        vertices_with_overlapping_domains[int(network_id)] = count
            else:
                domains = set(domain_type_list)
                for network_id, network in self.prot_inst.networks.items():
                    count = 0
                    for partner in network:
                        partner_domains = set(partner.domain_matches_dict.keys())
                        if domains.intersection(partner_domains) == domains:
                            count += 1
                            if network_id == 0:
                                self.partners_with_domain_in_real_network.add(partner)
                    vertices_with_overlapping_domains[int(network_id)] = count
            self.vertex_with_domain_in_real_network = vertices_with_overlapping_domains.get(0, 0)
            num_network_more_equal_real = len(
                [v for v in list(vertices_with_overlapping_domains.values())[1:] if v >= self.vertex_with_domain_in_real_network]
            )
            rand_network_mean = np.mean(list(vertices_with_overlapping_domains.values())[1:]) if num_rand_networks > 0 else 0
            rand_network_std = np.std(list(vertices_with_overlapping_domains.values())[1:]) if num_rand_networks > 0 else 0
            self.DomainEnrichment_pvalue = num_network_more_equal_real / num_rand_networks if num_rand_networks > 0 else 1.0
            if rand_network_std == 0:
                self.DomainEnrichment_zscore = self.vertex_with_domain_in_real_network - rand_network_mean
            else:
                self.DomainEnrichment_zscore = (
                    self.vertex_with_domain_in_real_network - rand_network_mean
                ) / rand_network_std

    def _set_cons_problem(self):
        self.qfo_RLC = None
        self.qfo_RLCvar = None
        self.vertebrates_RLC = None
        self.vertebrates_RLCvar = None
        self.mammalia_RLC = None
        self.mammalia_RLCvar = None
        self.metazoa_RLC = None
        self.metazoa_RLCvar = None

class DomainType(BaseDomainType):
    def __init__(self, domain_id):
        super().__init__(domain_id)
        self.dmi_types = []

class DomainInterfaceMatch(BaseDomainInterfaceMatch):
    def __init__(self, slim_id, domain_interface, domain_matches):
        super().__init__(domain_interface, domain_matches)
        self.slim_id = slim_id

class DMIType:
    def __init__(self, slim_id):
        self.slim_id = slim_id
        self.domain_interfaces = []

class DMIMatch:
    def __init__(self, slim_protein, domain_protein, slim_match, domain_interface_match):
        self.slim_protein = slim_protein
        self.domain_protein = domain_protein
        self.slim_match = slim_match
        self.domain_interface_match = domain_interface_match
        self.score = None
        self.missing_feature = None

class InterfaceHandling(BaseInterfaceHandling):
    def __init__(self, prot_path, slim_type_file, dmi_type_file, smart_domain_types_file, pfam_domain_types_file, smart_domain_matches_json_file, pfam_domain_matches_json_file, features_path, PPI_file=None, network_path=None):
        super().__init__(prot_path, PPI_file)
        self.slim_types_dict = {}
        self.dmi_types_dict = {}
        self.slim_type_file = slim_type_file
        self.dmi_type_file = dmi_type_file
        self.smart_domain_types_file = smart_domain_types_file
        self.pfam_domain_types_file = pfam_domain_types_file
        self.smart_domain_matches_json_file = smart_domain_matches_json_file
        self.pfam_domain_matches_json_file = pfam_domain_matches_json_file
        self.features_path = features_path
        self.network_path = network_path if network_path is not None else self.features_path

    def load_sequences_from_dict(self, sequences_dict):
        for protein_id, sequence in sequences_dict.items():
            prot_inst = Protein(protein_id)
            prot_inst.sequence = sequence
            self.proteins_dict[protein_id] = prot_inst

    def set_protein_pairs(self, ppi_pairs):
        for pair in ppi_pairs:
            sorted_pair = tuple(sorted(pair))
            self.protein_pairs_dict[sorted_pair] = ProteinPair(sorted_pair[0], sorted_pair[1])

    def read_in_proteins(self):
        file_names = [file_name for file_name in glob.glob(self.prot_path + '/*')]
        for file_name in file_names:
            with open(file_name, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            for line in lines:
                if line[0] == '>':
                    protein_id = line[1:]
                    prot_inst = Protein(protein_id)
                    self.proteins_dict[protein_id] = prot_inst
                else:
                    self.proteins_dict[protein_id].sequence = line
        print(f"{len(self.proteins_dict)} proteins read in.")

    def read_in_slim_types(self):
        with open(self.slim_type_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines[5:]:
            tab = line.split('\t')
            slim_id = tab[0]
            slim_type_inst = SLiMType(slim_id)
            slim_type_inst.name = tab[1]
            slim_type_inst.regex = tab[4].replace('"', '')
            slim_type_inst.probability = str(tab[5])
            self.slim_types_dict[slim_id] = slim_type_inst
        print(f"{len(self.slim_types_dict)} SLiM types read in.")

    def read_in_DMI_types(self):
        with open(self.dmi_type_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines[1:]:
            tab = line.split('\t')
            if tab[6] == '1':
                slim_id = tab[0]
                domain_id = tab[2]
                if domain_id not in self.domain_types_dict:
                    self.domain_types_dict[domain_id] = DomainType(domain_id)
                domain_id2 = ''
                domain_count = int(tab[5])
                domain_count2 = ''
                DMI_type_inst = DMIType(slim_id)
                if slim_id not in self.dmi_types_dict.keys():
                    self.dmi_types_dict[slim_id] = DMI_type_inst
                if (len(tab) < 11) or (len(tab) > 10 and tab[11] == ''):
                    domain_interface_inst = DomainInterface()
                    domain_interface_inst.domain_dict[domain_id] = int(domain_count)
                    self.dmi_types_dict[slim_id].domain_interfaces.append(domain_interface_inst)
                    self.domain_types_dict[domain_id].dmi_types.append(self.dmi_types_dict[slim_id])
                elif (len(tab) > 7) & (tab[11] == '1'):
                    domain_id2 = tab[7]
                    domain_count2 = tab[10]
                    domain_interface_inst = DomainInterface()
                    domain_interface_inst.domain_dict[domain_id] = int(domain_count)
                    domain_interface_inst.domain_dict[domain_id2] = int(domain_count2)
                    if domain_id2 not in self.domain_types_dict:
                        self.domain_types_dict[domain_id2] = DomainType(domain_id2)
                    self.dmi_types_dict[slim_id].domain_interfaces.append(domain_interface_inst)
                    self.domain_types_dict[domain_id2].dmi_types.append(self.dmi_types_dict[slim_id])
                elif (len(tab) > 7) & (tab[11] == '0'):
                    continue
        print(f"{len(self.dmi_types_dict)} DMI types read in.")

    def read_in_domain_types(self):
        for domain_types_file in [self.smart_domain_types_file, self.pfam_domain_types_file]:
            with open(domain_types_file, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            for line in lines[2:]:
                tab = line.split('\t')
                domain_id = tab[1]
                if domain_id in self.domain_types_dict:
                    self.domain_types_dict[domain_id].name = tab[0]
                    self.domain_types_dict[domain_id].descr = tab[2]
                    self.domain_types_dict[domain_id].DomainFreqbyProtein = tab[3]
                    self.domain_types_dict[domain_id].DomainFreqinProteome = tab[4]
                    if domain_id[:2] == 'PF':
                        self.domain_types_dict[domain_id].source = 'PFAM'
                    elif domain_id[:2] == 'SM':
                        self.domain_types_dict[domain_id].source = 'SMART'
            print(f"{domain_types_file} read in.")
        print(f"{len(self.domain_types_dict)} read in.")

    def read_in_domain_matches(self):
        if len(self.domain_types_dict) == 0:
            self.read_in_domain_types()
        for domain_matches_json_file in [self.smart_domain_matches_json_file, self.pfam_domain_matches_json_file]:
            with open(domain_matches_json_file, 'r') as f:
                data = json.load(f)
            for result in data['results']:
                protein_id = result['metadata']['accession']
                if protein_id in self.proteins_dict:
                    self.proteins_dict[protein_id].name = result['metadata']['name']
                    for domain_match_id in result['entry_subset']:
                        for domain_match in domain_match_id['entry_protein_locations']:
                            domain_id = domain_match['model']
                            score = domain_match['score']
                            if domain_id in self.domain_types_dict:
                                for fragment in domain_match['fragments']:
                                    start = fragment['start']
                                    end = fragment['end']
                                    domain_match_inst = DomainMatch(domain_id, start, end)
                                    domain_match_inst.evalue = score
                                    if domain_id not in self.proteins_dict[protein_id].domain_matches_dict.keys():
                                        self.proteins_dict[protein_id].domain_matches_dict[domain_id] = []
                                    self.proteins_dict[protein_id].domain_matches_dict[domain_id].append(domain_match_inst)

    def read_in_networks(self, prot_set=None):
        file_names = [file_name for file_name in glob.glob(self.network_path + '/Protein_networks/*')] if self.network_path else []
        for file_name in file_names:
            prot_file = file_name.split('/')[-1]
            prot_id = prot_file.split('_')[0]
            if prot_set is not None and prot_id not in prot_set:
                continue
            if prot_id in self.proteins_dict:
                with open(file_name, 'r') as file:
                    lines = [line.strip() for line in file.readlines()]
                for line in lines[1:]:
                    tabs = line.split('\t')
                    self.proteins_dict[prot_id].networks[int(tabs[0])] = []
                    if '|' in tabs[1]:
                        partners = tabs[1].split('|')
                        for partner in partners:
                            if partner in self.proteins_dict:
                                self.proteins_dict[prot_id].networks[int(tabs[0])].append(self.proteins_dict[partner])
                    else:
                        if tabs[1] in self.proteins_dict:
                            self.proteins_dict[prot_id].networks[int(tabs[0])].append(self.proteins_dict[tabs[1]])
                    self.proteins_dict[prot_id].network_degree = len(self.proteins_dict[prot_id].networks.get(0, []))

    def create_slim_matches_all_proteins(self):
        for prot_id, prot_inst in self.proteins_dict.items():
            for slim_id, dmi_type_inst in self.dmi_types_dict.items():
                slim_type_inst = self.slim_types_dict[slim_id]
                prot_inst.create_slim_matches(dmi_type_inst, slim_type_inst)

    def read_in_features_all_proteins(self):
        if not self.features_path:
            return
        for prot_id, prot_inst in self.proteins_dict.items():
            prot_inst.read_in_features(self.features_path)

    def calculate_features_scores_all_proteins(self):
        for prot_id, prot_inst in self.proteins_dict.items():
            prot_inst.calculate_features_scores()

    def find_DMI_matches(self):
        for protpair, protpair_inst in self.protein_pairs_dict.items():
            protein_pairs = [(protpair[0], protpair[1]), (protpair[1], protpair[0])]
            for protein_pair in protein_pairs:
                prot1_domain_matches_dict = self.proteins_dict[protein_pair[0]].domain_matches_dict
                prot2_slim_matches_dict = self.proteins_dict[protein_pair[1]].slim_matches_dict
                unique_slim_ids = set()
                for domain_id in prot1_domain_matches_dict.keys():
                    slim_id_list = [dmi_type.slim_id for dmi_type in self.domain_types_dict[domain_id].dmi_types]
                    slim_id_match_list = set(
                        list(filter(lambda slim_id: slim_id in prot2_slim_matches_dict, slim_id_list))
                    )
                    unique_slim_ids = unique_slim_ids.union(slim_id_match_list)
                for slim_id_match in unique_slim_ids:
                    domain_interface_list = self.dmi_types_dict[slim_id_match].domain_interfaces
                    for domain_interface in domain_interface_list:
                        cognate_domains = list(domain_interface.domain_dict.keys())
                        if set(cognate_domains).intersection(set(prot1_domain_matches_dict.keys())) == set(cognate_domains):
                            if len(cognate_domains) == 1:
                                domain_interface_match_inst = DomainInterfaceMatch(
                                    slim_id_match, domain_interface, [prot1_domain_matches_dict[cognate_domains[0]]]
                                )
                                for slim_match in prot2_slim_matches_dict[slim_id_match]:
                                    DMIMatch_inst = DMIMatch(
                                        protein_pair[1], protein_pair[0], slim_match, domain_interface_match_inst
                                    )
                                    if slim_id_match not in protpair_inst.dmi_matches_dict.keys():
                                        protpair_inst.dmi_matches_dict[slim_id_match] = []
                                    protpair_inst.dmi_matches_dict[slim_id_match].append(DMIMatch_inst)
                            elif len(cognate_domains) == 2:
                                domain_match1 = prot1_domain_matches_dict[cognate_domains[0]]
                                domain_match2 = prot1_domain_matches_dict[cognate_domains[1]]
                                domain_interface_match_inst = DomainInterfaceMatch(
                                    slim_id_match, domain_interface, [domain_match1, domain_match2]
                                )
                                for slim_match in prot2_slim_matches_dict[slim_id_match]:
                                    DMIMatch_inst = DMIMatch(
                                        protein_pair[1], protein_pair[0], slim_match, domain_interface_match_inst
                                    )
                                    if slim_id_match not in protpair_inst.dmi_matches_dict.keys():
                                        protpair_inst.dmi_matches_dict[slim_id_match] = []
                                    protpair_inst.dmi_matches_dict[slim_id_match].append(DMIMatch_inst)

