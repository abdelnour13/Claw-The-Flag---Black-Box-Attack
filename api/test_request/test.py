import requests
from tabulate import tabulate

if __name__ == '__main__':

    URL = "https://api-send.ctf.clawtheflag.com/predict" # "http://0.0.0.0:8000/predict"

    params = [
        ('threshold', 0.6),
        ('k', 10)
    ]

    documents = [
        {
            "description" : "Ontologies have been known for their powerful semantic representation of knowledge. However, ontologies cannot automatically evolve to reflect updates that occur in respective domains. To address this limitation, researchers have called for automatic ontology generation from unstructured text corpus. Unfortunately, systems that aim to generate ontologies from unstructured text corpus are domain-specific and require manual intervention. In addition, they suffer from uncertainty in creating concept linkages and difficulty in finding axioms for the same concept. Knowledge Graphs (KGs) has emerged as a powerful model for the dynamic representation of knowledge. However, KGs have many quality limitations and need extensive refinement. This research aims to develop a novel domain-independent automatic ontology generation framework that converts unstructured text corpus into domain consistent ontological form. The framework generates KGs from unstructured text corpus as well as refine and correct them to be consistent with domain ontologies. The power of the proposed automatically generated ontology is that it integrates the dynamic features of KGs and the quality features of ontologies",
        }
    ]

    body = {
        "documents" : documents
    }

    response = requests.post(URL, params=params, json=body)
    keywords = response.json()

    print(tabulate(keywords[0], tablefmt="double_grid", headers="keys"))