import re
from split_words import Splitter
# import spacy

class DEProcessor:
    def __init__(self):
        self.compound_splitter = Splitter()
        # self.lemmatizer = spacy.load("de_core_news_md")
        
    # def lowercase_splitcompound_lemmatize(self, text: str):
    #     text = text.lower()
    #     tokens = re.split(r"\s+", text)

    #     new_tokens = []
    #     for token in tokens:
    #         _l = self.compound_splitter.split_compound(token)[0]
    #         if _l[0] <= 0:
    #             new_tokens.append(token)
    #         else:
    #             new_tokens.extend(_l[1:])

    #     new_text = ' '.join([token for token in new_tokens])
            
    #     tokens = self.lemmatizer(new_text)

    #     new_text = ' '.join([token.lemma_.lower() for token in tokens])

    #     return new_text

    def lowercase_splitcompound(self, text: str):
        text = text.lower()
        tokens = re.split(r"\s+", text)

        new_tokens = []
        for token in tokens:
            _l = self.compound_splitter.split_compound(token)[0]
            if _l[0] <= 0:
                new_tokens.append(token)
            else:
                new_tokens.extend(_l[1:])

        new_text = ' '.join([token.lower() for token in new_tokens])

        return new_text

if __name__ == "__main__":
    p = DEProcessor()

    text = 'Die Verträge wurden aufgrund einer fristlosen Kündigung des Arbeitsverhältnisses nach § 626 BGB wegen schwerwiegender Pflichtverletzungen beendet.Der Schadensersatzanspruch des Arbeitnehmers besteht weiterhin.'
    
    print(p.lowercase_splitcompound_lemmatize(text))
    