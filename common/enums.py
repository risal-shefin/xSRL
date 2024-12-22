from enum import Enum

# Counterfactual Action Methods
class CtfActionMethodEnum(Enum):
  RiskyOnce = 0
  RiskyAlways = 1

class SummaryMethodEnum(Enum):
  Average = 0
  Max = 1
  Min = 2

class PolicyEnum(Enum):
  TaskPolicy = 0
  SafetyPolicy = 1
