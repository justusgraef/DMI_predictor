"""
Core data structures for protein interaction interfaces (ported from original scripts).
"""
import glob
import json

class Protein:
    def __init__(self, protein_id):
        self.protein_id = protein_id
        self.sequence = ''
        self.domain_matches_dict = {}
        self.name = ''

class ProteinPair:
    def __init__(self, proteinA, proteinB):
        self.proteinA = proteinA
        self.proteinB = proteinB

class DomainType:
    def __init__(self, domain_id):
        self.domain_id = domain_id
        self.name = ''
        self.source = ''
        self.DomainFreqbyProtein = None
        self.DomainFreqinProteome = None
        self.dmi_types = []
        self.descr = ''

class DomainMatch:
    def __init__(self, domain_id, start, end):
        self.domain_id = domain_id
        self.start = start
        self.end = end
        self.evalue = None

class DomainInterface:
    def __init__(self):
        self.domain_dict = {}

class DomainInterfaceMatch:
    def __init__(self, domain_interface, domain_matches):
        self.domain_interface = domain_interface
        self.domain_matches = domain_matches

class InterfaceHandling:
    def __init__(self, prot_path, PPI_file=None):
        if PPI_file is not None:
            self.PPI_file = PPI_file
        self.proteins_dict = {}
        self.domain_types_dict = {}
        self.known_PPIs = set()
        self.protein_pairs_dict = {}
        self.prot_path = prot_path

    def read_in_proteins(self, only_canonical=True):
        if only_canonical is False:
            file_names = [file_name for file_name in glob.glob(self.prot_path + '/*')]
        else:
            file_names = [file_name for file_name in glob.glob(self.prot_path + '/*') if '-' not in file_name]
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

    def read_in_domain_types(self, domain_types_file):
        with open(domain_types_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines[2:]:
            tab = line.split('\t')
            name = tab[0]
            domain_id = tab[1]
            descr = tab[2] if len(tab) > 2 else ''
            if domain_id not in self.domain_types_dict:
                self.domain_types_dict[domain_id] = DomainType(domain_id)
            self.domain_types_dict[domain_id].name = name
            self.domain_types_dict[domain_id].descr = descr
            if domain_id[:2] == 'PF':
                self.domain_types_dict[domain_id].source = 'PFAM'
            elif domain_id[:2] == 'SM':
                self.domain_types_dict[domain_id].source = 'SMART'

    def read_in_known_PPIs(self):
        with open(self.PPI_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            tab = line.split('\t')
            PPI_instance = sorted(list([tab[0], tab[1]]))
            self.known_PPIs.add(tuple(PPI_instance))
        print(f"{len(self.known_PPIs)} PPIs read in.")

    def read_in_domain_matches(self, domain_matches_json_file):
        with open(domain_matches_json_file) as f:
            data = json.load(f)
        for result in data['results']:
            protein_id = result['metadata']['accession']
            if protein_id in self.proteins_dict:
                self.proteins_dict[protein_id].name = result['metadata']['name']
                for domain_match_id in result['entry_subset']:
                    for domain_match in domain_match_id['entry_protein_locations']:
                        domain_id = domain_match['model']
                        score = domain_match['score']
                        for fragment in domain_match['fragments']:
                            start = fragment['start']
                            end = fragment['end']
                            domain_match_inst = DomainMatch(domain_id, start, end)
                            domain_match_inst.evalue = score
                            if domain_id not in self.proteins_dict[protein_id].domain_matches_dict.keys():
                                self.proteins_dict[protein_id].domain_matches_dict[domain_id] = []
                            self.proteins_dict[protein_id].domain_matches_dict[domain_id].append(domain_match_inst)
