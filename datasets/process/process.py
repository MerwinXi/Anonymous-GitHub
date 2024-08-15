import collections.abc


class Process:
    """
    Compose multiple processes sequentially.

    Args:
        processes (Sequence[dict | callable]): Sequence of process objects or
            config dicts to be composed.
        cfg: Configuration object to be passed to the processes if necessary.
    """

    def __init__(self, processes, cfg):
        assert isinstance(processes, collections.abc.Sequence), "Processes should be a sequence."
        self.processes = [self._build_process(process, cfg) for process in processes]

    @staticmethod
    def _build_process(process, cfg):
        """
        Builds a process either from a dict or callable.

        Args:
            process (dict | callable): The process configuration or callable.
            cfg: The configuration object.

        Returns:
            callable: A callable process.
        """
        if isinstance(process, dict):
            return build_from_cfg(process, default_args=dict(cfg=cfg))
        elif callable(process):
            return process
        else:
            raise TypeError('Process must be callable or a dictionary')

    def __call__(self, data):
        """
        Apply processes sequentially to the input data.

        Args:
            data (dict): A dictionary containing the data to process.

        Returns:
            dict: Processed data, or None if any process returns None.
        """
        for process in self.processes:
            data = process(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        process_descriptions = "\n".join([f'    {process}' for process in self.processes])
        return f"{self.__class__.__name__}(\n{process_descriptions}\n)"
