from tensorboard.backend.event_processing import event_accumulator
import sys, os
# <current_directory>/../
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../")
from utils.directory import process_logfiles

# 
def average(dir:str, scalar_names:list,
            start_step:int=None, end_step:int=None):
    if start_step is None:
        start_step = -float('inf')
    if end_step is None:
        end_step = float('inf')
        
    # For extracting and saving scalar values
    class ScalarExtractor():
        def __init__(self):
            self.scalar_dict = None
        def get_scalars_within_steps(self, ea, scalar_names, start_step, end_step,
            **kwargs) -> dict:
            scalar_dict = dict()
            for scalar_name in scalar_names:
                scalar_list = ea.Scalars(scalar_name)
                scalar_dict[scalar_name] = []
                # loop for step
                for scalar in scalar_list:
                    if start_step <= scalar.step <= end_step:
                        scalar_dict[scalar_name].append(scalar.value)
            self.scalar_dict = scalar_dict
            return scalar_dict
            
    scalarextractor = ScalarExtractor()
    scalar_dict = process_logfiles(dir=dir, 
                                   func=scalarextractor.get_scalars_within_steps, 
                                   func_kwargs={"scalar_names":scalar_names,
                                                "start_step": start_step, 
                                                "end_step": end_step,
                                                })
    mean_dict = dict()
    for scalar_name, scalar_values in scalarextractor.scalar_dict.items():
        scalar_values:list
        mean_dict[scalar_name] = sum(scalar_values) / len(scalar_values)
    return mean_dict


def main():
    pass

if __name__ == "__main__":
    scalar_names = ["reward/dead",
                    "reward/direction",
                    "reward/progress",
                    "reward/safe",
                    "reward/time",
                    "reward/total",
                    ]
    for i in (0,1,2,3,4,5):
        print(f"PPO_{i}")
        directory = f"/home/control03/hojun/reward_elements/PPO_{i}"
        mean_dict = average(dir=directory, 
                            scalar_names=scalar_names,
                            start_step=1_500_000, end_step=2_000_000,
                            )
        
        for name, value in mean_dict.items():
            if 'progress' in name:
                print(f"{name}: {value}")
        print()