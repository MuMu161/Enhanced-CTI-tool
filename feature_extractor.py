import re
import nltk


class FeatureExtractor:

    def __init__(self):
        """
        Ensure required NLTK data is available.
        """
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    def extract(self, text):
        """
        Extract structured features from CTI text.
        """

        if not text:
            return {
                "word_count": 0,
                "noun_density": 0,
                "verb_density": 0,
                "cve_count": 0,
                "ip_count": 0,
            }

        tokens = nltk.word_tokenize(text)

        # IMPORTANT: explicitly specify lang='eng'
        pos_tags = nltk.pos_tag(tokens, lang="eng")

        noun_count = sum(1 for word, pos in pos_tags if pos.startswith("NN"))
        verb_count = sum(1 for word, pos in pos_tags if pos.startswith("VB"))

        cve_matches = re.findall(r"CVE-\d{4}-\d+", text)
        ip_matches = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)

        features = {
            "word_count": len(tokens),
            "noun_density": noun_count / len(tokens) if tokens else 0,
            "verb_density": verb_count / len(tokens) if tokens else 0,
            "cve_count": len(cve_matches),
            "ip_count": len(ip_matches),
        }

        return features
