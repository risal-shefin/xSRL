from enum import Enum
import argparse

# Counterfactual Action Methods
class CtfActionMethodEnum(Enum):
  RiskyOnce = 0
  RiskyAlways = 1
  
  @staticmethod
  def ctf_action_method_type(arg_value):
    try:
      # Look up the enum member by name
      return CtfActionMethodEnum[arg_value]
    except KeyError:
      raise argparse.ArgumentTypeError(
        f"Invalid value for --ctf_action_method: '{arg_value}'. Valid options are 'RiskyOnce' or 'RiskyAlways'."
      )

class SummaryMethodEnum(Enum):
  Average = 0
  Max = 1
  Min = 2

class PolicyEnum(Enum):
  TaskPolicy = 0
  SafetyPolicy = 1
