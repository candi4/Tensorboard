import os
from tensorboard.backend.event_processing import event_accumulator


# Use function for tensorboard log files
def process_logfiles(dir, *, func=None, func_kwargs=dict()) -> list:
    """
    @param dir: Directory to be searched
    @param func: Function to be used with log files. It is recommended to have `**kwargs` as function parameter.
        Using parameter: filename, ea
    @param func_kwargs: For func
    """
    file_list = []
    for root, dirs, files in os.walk(dir):
        for item in files:
            filename = os.path.join(root, item)
            ea = event_accumulator.EventAccumulator(filename)
            ea.Reload()
            if len(ea.Tags()['scalars']) > 0:
                file_list.append(filename)
                if func is not None:
                    kwargs_ = func_kwargs.copy()
                    kwargs_.update({'filename':filename,
                                    'ea': ea})
                    func(**kwargs_)
    return file_list


if __name__ == "__main__":
    def f(file=None, **kwargs):
        print(file)
    file_list = process_logfiles(dir="/home/control03/hojun/reward_elements/PPO_0", func=f)
    print(file_list)