from common.enums import SummaryMethodEnum


class DictionarySummaryModel:
  name: str
  dictionary: dict
  summary_method: SummaryMethodEnum

  def __init__(self, name: str, dictionary: dict, summary_method: SummaryMethodEnum):
    self.name = name
    self.dictionary = dictionary
    self.summary_method = summary_method