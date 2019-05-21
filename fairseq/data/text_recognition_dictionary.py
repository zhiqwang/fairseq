from fairseq.data import Dictionary


class TextRecognitionDictionary(Dictionary):
    """
    Dictionary for Text Recognition tasks. This extends Dictionary by
    adding the blank symbol.
    """
    def __init__(self, blank='<blank>'):
        self.blank_word = blank
        self.symbols = []
        self.count = []
        self.indices = {}
        self.blank_index = self.add_symbol(blank)
        self.nspecial = len(self.symbols)

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.blank_index
